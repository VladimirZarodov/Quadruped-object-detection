"""CV модуль для симулятора (Isaac Sim): YOLO + ORB matching (верификация / fallback)

Входные данные: RGB np.ndarray + Depth np.ndarray (в метрах)
Выходные данные: список Detection с 3D-позицией в системе координат камеры

Использование в контроллере:
    detector = ObjectDetector()
    detections = detector.detect(camera_data["image"], camera_data["depth"])
    best = detector.pick_target(detections, current_label)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"

# COCO-имя -> внутренний object_id (0-4)
OBJECT_CLASSES: Dict[str, int] = {
    "cup": 0,
    "laptop": 1,
    "bottle": 2,
    "chair": 3,
    "backpack": 4,
}

# Русские подписи для логов
LABEL_RU: Dict[str, str] = {
    "cup": "Чашка",
    "laptop": "Ноутбук",
    "bottle": "Бутылка",
    "chair": "Стул",
    "backpack": "Рюкзак",
}

_LOG = logging.getLogger(__name__)


def resource_usage_line() -> str:
    """
    Краткая строка: RAM системы / процесса и VRAM GPU (если есть).
    Используется в логах detect() и в тестах.
    """
    parts: List[str] = []

    try:
        import psutil  # type: ignore[import-untyped]

        vm = psutil.virtual_memory()
        parts.append(
            f"RAM системы {vm.used / (1024**3):.1f}/{vm.total / (1024**3):.1f} ГБ ({vm.percent:.0f}%)"
        )
        rss = psutil.Process().memory_info().rss / (1024**3)
        parts.append(f"RAM процесса {rss:.2f} ГБ")
    except Exception:
        meminfo_p = Path("/proc/meminfo")
        status_p = Path("/proc/self/status")
        if meminfo_p.is_file() and status_p.is_file():
            try:
                total_kb = avail_kb = None
                for line in meminfo_p.read_text().splitlines():
                    if line.startswith("MemTotal:"):
                        total_kb = int(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = int(line.split()[1])
                if total_kb:
                    used_gb = (total_kb - (avail_kb or 0)) / (1024**2)
                    parts.append(
                        f"RAM системы ~{used_gb:.1f}/{total_kb / (1024**2):.1f} ГБ"
                    )
                for line in status_p.read_text().splitlines():
                    if line.startswith("VmRSS:"):
                        rss_kb = int(line.split()[1])
                        parts.append(f"RAM процесса {rss_kb / (1024**2):.2f} ГБ")
                        break
            except OSError:
                parts.append("RAM: n/d")
        else:
            parts.append("RAM: n/d")

    try:
        import torch

        if torch.cuda.is_available():
            dev = torch.cuda.current_device()
            u = torch.cuda.memory_allocated(dev) / (1024**3)
            tot = torch.cuda.get_device_properties(dev).total_memory / (1024**3)
            parts.append(f"VRAM {u:.2f}/{tot:.1f} ГБ (GPU {dev})")
        else:
            parts.append("VRAM: нет CUDA (CPU)")
    except Exception:
        parts.append("VRAM: n/d")

    return " | ".join(parts)


# Модель по умолчанию (лежит в корне репозитория)
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = _REPO_ROOT / "yolo26m.pt"
FALLBACK_MODEL_NAME = "yolo26m.pt"

# Размер входа YOLO по умолчанию для симулятора (меньше 640 → быстрее, точность обычно достаточна)
DEFAULT_YOLO_IMGSZ: int = 416

# Параметры камеры по умолчанию для симулятора
# Горизонтальный FOV = 87 градусов (стандартная настройка камеры в симуляторе)
DEFAULT_HFOV_DEG: float = 87.0
# При другом разрешении -> detector.update_intrinsics_from_image(h, w, hfov)
DEFAULT_FX: float = 606.0
DEFAULT_FY: float = 606.0
DEFAULT_CX: float = 320.0
DEFAULT_CY: float = 240.0

# ORB matching
_ORB_MIN_GOOD_MATCHES = 8      # порог хороших совпадений для уверенной классификации
_ORB_LOWE_RATIO = 0.75         # Lowe's ratio test
_ORB_MIN_RATIO_SCORE = 0.04    # минимальное отношение good/total для репорта

# Усреднение глубины: окно (2k+1)×(2k+1) вокруг центра bbox
_DEPTH_HALF_WIN = 5

# Объект считается "достигнутым" при расстоянии <= 0.5 м
REACH_DISTANCE_M: float = 0.5

@dataclass
class Detection:
    """Результат детекции одного объекта."""
    label: str                    # "cup" | "laptop" | "bottle" | "chair" | "backpack"
    object_id: int                # 0-4
    confidence: float             # основная уверенность (YOLO conf или ORB score)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2) в пикселях
    distance_m: float             # глубина по оси Z, метры (0 = неизвестно)
    xyz_camera: np.ndarray        # [x, y, z] в СК камеры, метры
    source: str = "yolo"          # "yolo" | "yolo+orb" | "orb_corrected" | "orb"
    yolo_confidence: Optional[float] = None  # conf бокса YOLO; None если только ORB fallback
    orb_score: Optional[float] = None        # ratio good/len(des) из classify; None если ORB не прошёл пороги

    @property
    def is_reached(self) -> bool:
        """True, если объект находится в зоне достижения (<= REACH_DISTANCE_M)"""
        return 0 < self.distance_m <= REACH_DISTANCE_M

    def __repr__(self) -> str:
        d = f"{self.distance_m:.2f}m" if self.distance_m > 0 else "??m"
        y = f"{self.yolo_confidence:.2f}" if self.yolo_confidence is not None else "-"
        o = f"{self.orb_score:.3f}" if self.orb_score is not None else "-"
        return (
            f"Detection({self.label}, id={self.object_id}, "
            f"conf={self.confidence:.2f}, yolo={y}, orb={o}, dist={d}, src={self.source})"
        )


# ORB feature matcher


class _ORBMatcher:
    """Сопоставление эталонных изображений из /assets/ с кропом по ORB + BFMatcher"""

    def __init__(self, assets_dir: Path) -> None:
        self._orb = cv2.ORB_create(nfeatures=500)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._refs: Dict[str, Tuple] = {}
        self._load_references(assets_dir)

    def _load_references(self, assets_dir: Path) -> None:
        loaded = []
        for label in OBJECT_CLASSES:
            path = self._find_asset(assets_dir, label)
            if path is None:
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            kp, des = self._orb.detectAndCompute(gray, None)
            if des is not None and len(des) >= 4:
                self._refs[label] = (kp, des, len(kp))
                loaded.append(label)
        if loaded:
            print(f"[ORBMatcher] загружены эталоны: {loaded}")
        else:
            print("[ORBMatcher] Предупреждение: эталонные изображения не найдены")

    @staticmethod
    def _find_asset(assets_dir: Path, label: str) -> Optional[Path]:
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
            p = assets_dir / f"{label}{ext}"
            if p.exists():
                return p
        return None

    def classify(
        self, crop_bgr: np.ndarray
    ) -> Optional[Tuple[str, int, float]]:
        """
        Определить класс кропа по ORB
        Возвращает (label, object_id, score) или None, если совпадений недостаточно
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(des) < 4:
            return None

        best_label: Optional[str] = None
        best_good = 0

        for label, (ref_kp, ref_des, _) in self._refs.items():
            try:
                pairs = self._bf.knnMatch(ref_des, des, k=2)
            except cv2.error:
                continue
            good = [
                m for pair in pairs
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < _ORB_LOWE_RATIO * n.distance
            ]
            if len(good) > best_good:
                best_good = len(good)
                best_label = label

        if best_label is None or best_good < _ORB_MIN_GOOD_MATCHES:
            return None

        score = best_good / max(len(des), 1)
        if score < _ORB_MIN_RATIO_SCORE:
            return None

        return best_label, OBJECT_CLASSES[best_label], score

    def scores_all_labels(self, crop_bgr: np.ndarray) -> Dict[str, Tuple[int, float]]:
        """
        По каждому загруженному эталону: (число good-matches, ratio = good / len(descriptors_кропа)).
        Без порогов classify — чтобы смотреть «сырые» числа и подбирать пороги.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return {}
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        kp, des = self._orb.detectAndCompute(gray, None)
        if des is None or len(des) < 4:
            return {lbl: (0, 0.0) for lbl in self._refs}

        n_des = len(des)
        out: Dict[str, Tuple[int, float]] = {}
        for label, (_, ref_des, _) in self._refs.items():
            try:
                pairs = self._bf.knnMatch(ref_des, des, k=2)
            except cv2.error:
                out[label] = (0, 0.0)
                continue
            good = [
                m for pair in pairs
                if len(pair) == 2
                for m, n in [pair]
                if m.distance < _ORB_LOWE_RATIO * n.distance
            ]
            g = len(good)
            out[label] = (g, g / max(n_des, 1))
        return out

    @property
    def is_ready(self) -> bool:
        return len(self._refs) > 0


def _sample_depth(
    depth: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> float:
    """Медиана ненулевых пикселей глубины в центральном окне bbox"""
    h, w = depth.shape[:2]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    wx1 = max(cx - _DEPTH_HALF_WIN, x1, 0)
    wy1 = max(cy - _DEPTH_HALF_WIN, y1, 0)
    wx2 = min(cx + _DEPTH_HALF_WIN + 1, x2, w)
    wy2 = min(cy + _DEPTH_HALF_WIN + 1, y2, h)
    patch = depth[wy1:wy2, wx1:wx2].astype(np.float32)
    valid = patch[(patch > 0.05) & np.isfinite(patch)]
    return float(np.median(valid)) if valid.size > 0 else 0.0


def _pixel_to_xyz(
    px: int, py: int, depth_m: float,
    fx: float, fy: float, cx: float, cy: float,
) -> np.ndarray:
    """Обратная проекция пикселя + глубины -> 3D точка в СК камеры"""
    if depth_m <= 0.0:
        return np.zeros(3, dtype=np.float32)
    z = depth_m
    x = (px - cx) * z / fx
    y = (py - cy) * z / fy
    return np.array([x, y, z], dtype=np.float32)


class ObjectDetector:
    """
    Детектор объектов на кубах для симулятора

    Первичный метод: YOLO (COCO, классы cup/laptop/bottle/chair/backpack)
    Вторичный метод: ORB-matching по эталонам из /assets/ — верификация и fallback

    Параметры:
        model_path      — путь к .pt (по умолчанию yolo26n.pt из корня репо; для точности можно yolo26x.pt)
        assets_dir      — папка с эталонами (по умолчанию /assets/)
        fx, fy, cx, cy  — параметры камеры (по умолчанию для Isaac Sim 640×480, FOV 87°)
        yolo_conf       — порог уверенности YOLO (по умолчанию 0.1)
        use_orb_verify  — True: ORB проверяет каждый бокс YOLO
        use_orb_fallback— True: ORB ищет объект, если YOLO ничего не нашёл
        log_detections  — писать в logging INFO строки при каждом detect()
    """

    def __init__(
        self,
        *,
        model_path: Optional[str | Path] = None,
        assets_dir: Optional[str | Path] = None,
        fx: float = DEFAULT_FX,
        fy: float = DEFAULT_FY,
        cx: float = DEFAULT_CX,
        cy: float = DEFAULT_CY,
        yolo_conf: float = 0.1,
        yolo_imgsz: int = 640,
        use_orb_verify: bool = True,
        use_orb_fallback: bool = True,
        log_detections: bool = True,
    ) -> None:
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.yolo_conf = yolo_conf
        self.yolo_imgsz = yolo_imgsz
        self.use_orb_verify = use_orb_verify
        self.use_orb_fallback = use_orb_fallback
        self._log_detections = log_detections
        self._intrinsics_updated = False
        self.start_time = time.perf_counter()

        mp: str | Path
        if model_path is not None:
            mp = Path(model_path)
        elif DEFAULT_MODEL_PATH.exists():
            mp = DEFAULT_MODEL_PATH
        else:
            mp = FALLBACK_MODEL_NAME
        self._yolo = YOLO(str(mp))
        
        # Принудительно кладём модель на GPU (cuda:0), если он доступен
        import torch
        if torch.cuda.is_available():
            self._yolo.to("cuda:0")
            self._device = "0"
            print(f"[ObjectDetector] YOLO загружен: {mp} (на GPU: {torch.cuda.get_device_name(0)})")
        else:
            self._device = "cpu"
            print(f"[ObjectDetector] YOLO загружен: {mp} (на CPU)")

        _assets = Path(assets_dir) if assets_dir else ASSETS_DIR
        self._orb = _ORBMatcher(_assets)

    def _yolo_predict_kw(self, conf: float) -> Dict[str, object]:
        """Общие kwargs для YOLO predict (GPU только если модель на CUDA)."""
        kw: Dict[str, object] = {
            "imgsz": self.yolo_imgsz,
            "conf": conf,
            "verbose": False,
        }
        if getattr(self, "_device", "cpu") == "0":
            kw["device"] = 0
        return kw

    def update_intrinsics_from_image(
        self, h: int, w: int, hfov_deg: float = DEFAULT_HFOV_DEG
    ) -> None:
        """Авто-оценка параметров камеры по размеру кадра и горизонтальному FOV"""
        hfov_rad = hfov_deg * np.pi / 180.0
        fx = (w / 2.0) / np.tan(hfov_rad / 2.0)
        self.fx = fx
        self.fy = fx
        self.cx = w / 2.0
        self.cy = h / 2.0
        self._intrinsics_updated = True

    def _maybe_update_intrinsics(self, h: int, w: int) -> None:
        """Обновить параметры камеры один раз, если разрешение отличается от дефолта 640x480"""
        if not self._intrinsics_updated and (w != int(DEFAULT_CX * 2) or h != int(DEFAULT_CY * 2)):
            self.update_intrinsics_from_image(h, w)
            print(f"[ObjectDetector] интринсика обновлена для симулятора на разрешение {w}×{h}")

    def yolo_max_confidence_by_class(
        self, rgb: np.ndarray, conf_floor: float = 0.01
    ) -> Dict[str, float]:
        """
        Максимальный confidence YOLO по каждому классу из OBJECT_CLASSES.
        Отдельный прогон с низким порогом conf_floor — чтобы видеть «хвост» ниже yolo_conf.
        """
        h, w = rgb.shape[:2]
        self._maybe_update_intrinsics(h, w)
        results = self._yolo.predict(source=rgb, **self._yolo_predict_kw(conf_floor))
        r = results[0]
        best: Dict[str, float] = {c: 0.0 for c in OBJECT_CLASSES}
        if r.boxes is None:
            return best
        for box in r.boxes:
            cls_id = int(box.cls.item())
            label = str(r.names[cls_id])
            if label not in best:
                continue
            cf = float(box.conf.item())
            if cf > best[label]:
                best[label] = cf
        return best

    def orb_scores_full_frame(self, rgb: np.ndarray) -> Dict[str, Tuple[int, float]]:
        """ORB по полному кадру: для каждого эталона (good_matches, ratio)."""
        h, w = rgb.shape[:2]
        self._maybe_update_intrinsics(h, w)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return self._orb.scores_all_labels(bgr)

    def detect(
        self,
        rgb: np.ndarray,
        depth: Optional[np.ndarray] = None,
    ) -> List[Detection]:
        """
        Детектировать объекты на кадре

        Args:
            rgb:   H×W×3 uint8, формат RGB
            depth: H×W float32, глубина в метрах (None -> только 2D детекция)

        Returns:
            Список Detection, отсортированный по расстоянию (ближайшие первые)
        """
        h_img, w_img = rgb.shape[:2]
        self._maybe_update_intrinsics(h_img, w_img)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        t0 = time.perf_counter()

        detections: List[Detection] = []
        seen_labels: set = set()

        # --- YOLO ---
        results = self._yolo.predict(source=rgb, **self._yolo_predict_kw(self.yolo_conf))
        r = results[0]

        if r.boxes is not None:
            for box in r.boxes:
                cls_id = int(box.cls.item())
                label = str(r.names[cls_id])
                if label not in OBJECT_CLASSES:
                    continue

                conf = float(box.conf.item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].tolist())
                source = "yolo"
                yolo_confidence = conf
                orb_score: Optional[float] = None

                # ORB верификация класса
                if self.use_orb_verify and self._orb.is_ready:
                    crop = bgr[
                        max(y1, 0):min(y2, h_img),
                        max(x1, 0):min(x2, w_img),
                    ]
                    orb_result = self._orb.classify(crop)
                    if orb_result is not None:
                        orb_label, _, orb_s = orb_result
                        orb_score = orb_s
                        if orb_label != label:
                            label = orb_label
                            source = "orb_corrected"
                        else:
                            source = "yolo+orb"

                obj_id = OBJECT_CLASSES[label]
                px, py = (x1 + x2) // 2, (y1 + y2) // 2
                dist = 0.0
                xyz = np.zeros(3, dtype=np.float32)

                if depth is not None:
                    dist = _sample_depth(depth, x1, y1, x2, y2)
                    xyz = _pixel_to_xyz(
                        px, py, dist, self.fx, self.fy, self.cx, self.cy
                    )

                seen_labels.add(label)
                detections.append(Detection(
                    label=label,
                    object_id=obj_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    distance_m=dist,
                    xyz_camera=xyz,
                    source=source,
                    yolo_confidence=yolo_confidence,
                    orb_score=orb_score,
                ))

        # --- ORB fallback: поиск объектов, которые YOLO не нашёл ---
        if self.use_orb_fallback and self._orb.is_ready and depth is not None:
            orb_result = self._orb.classify(bgr)
            if orb_result is not None:
                orb_label, orb_id, orb_score = orb_result
                if orb_label not in seen_labels:
                    valid_d = depth[(depth > 0.05) & np.isfinite(depth)]
                    dist = float(np.median(valid_d)) if valid_d.size > 0 else 0.0
                    px, py = w_img // 2, h_img // 2
                    xyz = _pixel_to_xyz(
                        px, py, dist, self.fx, self.fy, self.cx, self.cy
                    )
                    detections.append(Detection(
                        label=orb_label,
                        object_id=orb_id,
                        confidence=orb_score,
                        bbox=(0, 0, w_img, h_img),
                        distance_m=dist,
                        xyz_camera=xyz,
                        source="orb",
                        yolo_confidence=None,
                        orb_score=orb_score,
                    ))

        # Сортировка: ближайшие первые, затем по убыванию уверенности
        detections.sort(
            key=lambda d: (d.distance_m if d.distance_m > 0 else 999.0, -d.confidence)
        )

        dt = max(time.perf_counter() - t0, 1e-9)
        fps_eff = 1.0 / dt
        elapsed_time = time.perf_counter() - self.start_time
        if self._log_detections and _LOG.isEnabledFor(logging.INFO):
            res = resource_usage_line()
            if detections:
                for d in detections:
                    ru = LABEL_RU.get(d.label, d.label)
                    ys = f"{d.yolo_confidence:.3f}" if d.yolo_confidence is not None else "—"
                    os_ = f"{d.orb_score:.3f}" if d.orb_score is not None else "—"
                    _LOG.info(
                        "[Секундомер: %.1f c] Найден класс %s (%s), id=%d, уверенность=%.3f, yolo=%s, orb=%s, "
                        "источник=%s, расстояние=%.2f м",
                        elapsed_time,
                        ru,
                        d.label,
                        d.object_id,
                        d.confidence,
                        ys,
                        os_,
                        d.source,
                        d.distance_m,
                    )
                _LOG.info("[Секундомер: %.1f c] Кадр: эфф. FPS=%.2f | %s", elapsed_time, fps_eff, res)
            else:
                _LOG.info("[Секундомер: %.1f c] Детекций нет | эфф. FPS=%.2f | %s", elapsed_time, fps_eff, res)

        return detections

    def pick_target(
        self,
        detections: List[Detection],
        target_label: str,
    ) -> Optional[Detection]:
        """Выбрать лучшую детекцию для целевого класса"""
        candidates = [d for d in detections if d.label == target_label]
        return candidates[0] if candidates else None

    def draw(
        self,
        rgb: np.ndarray,
        detections: List[Detection],
        target_label: Optional[str] = None,
    ) -> np.ndarray:
        """
        Нарисовать bbox и метки на кадре RGB
        Возвращает BGR изображение (для cv2.imshow)
        """
        vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            is_target = (target_label is not None and det.label == target_label)
            color = (0, 255, 0) if is_target else (200, 200, 0)
            thickness = 3 if is_target else 2
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
            dist_str = f"{det.distance_m:.2f}m" if det.distance_m > 0 else "??m"
            yc = f"{det.yolo_confidence:.2f}" if det.yolo_confidence is not None else "-"
            oc = f"{det.orb_score:.3f}" if det.orb_score is not None else "-"
            text = f"{det.label} y={yc} o={oc} {dist_str} [{det.source}]"
            cv2.putText(
                vis, text, (x1, max(y1 - 6, 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
            )
        return vis
