import csv
import numpy as np
import matplotlib.pyplot as plt
from isaacgym import terrain_utils


class Terrain:
    def __init__(
        self,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        border_size=0.0,
        terrain_length=20.0,
        terrain_width=12.0,
        mesh_type="trimesh",
        slope_treshold=1.5,
    ):
        self.type = mesh_type
        self.horizontal_scale = horizontal_scale
        self.vertical_scale = vertical_scale
        self.border_size = border_size
        self.cfg = self
        self.terrain_length = terrain_length
        self.terrain_width = terrain_width
        self.slope_treshold = slope_treshold
        self.wall_height = 1.0
        self.wall_thickness = 0.5

        if self.type == "none":
            return

        self.length_per_env_pixels = int(self.terrain_length / self.horizontal_scale)
        self.width_per_env_pixels = int(self.terrain_width / self.horizontal_scale)
        self.border = int(self.border_size / self.horizontal_scale)
        self.wall_height_px = int(self.wall_height / self.vertical_scale)
        self.wall_thickness_px = max(1, int(self.wall_thickness / self.horizontal_scale))

        self.tot_rows = self.width_per_env_pixels + 2 * self.border
        self.tot_cols = self.length_per_env_pixels + 2 * self.border

        self.height_field_raw = np.zeros((self.tot_rows, self.tot_cols), dtype=np.int16)

        terrain = terrain_utils.SubTerrain(
            "flat",
            width=self.width_per_env_pixels,
            length=self.length_per_env_pixels,
            vertical_scale=self.vertical_scale,
            horizontal_scale=self.horizontal_scale,
        )
        terrain.height_field_raw[:, :] = 0

        start_x = self.border
        end_x = self.border + self.width_per_env_pixels
        start_y = self.border
        end_y = self.border + self.length_per_env_pixels

        self.height_field_raw[start_x:end_x, start_y:end_y] = terrain.height_field_raw

        # walls around the terrain
        self.height_field_raw[start_x:start_x + self.wall_thickness_px, start_y:end_y] = self.wall_height_px
        self.height_field_raw[end_x - self.wall_thickness_px:end_x, start_y:end_y] = self.wall_height_px
        self.height_field_raw[start_x:end_x, start_y:start_y + self.wall_thickness_px] = self.wall_height_px
        self.height_field_raw[start_x:end_x, end_y - self.wall_thickness_px:end_y] = self.wall_height_px

        # internal wall
        inner_wall_length = 2.0
        inner_wall_thickness = 0.5

        inner_wall_length_px = max(1, int(inner_wall_length / self.horizontal_scale))
        inner_wall_thickness_px = max(1, int(inner_wall_thickness / self.horizontal_scale))

        # place it approximately in the center of the flat area
        center_x = (start_x + end_x) // 2 + 5
        center_y = (start_y + end_y) // 2 - 15

        # vertical wall (long in y direction, thick in x direction)
        wall_x0 = center_x - inner_wall_thickness_px // 2
        wall_x1 = wall_x0 + inner_wall_thickness_px

        wall_y0 = center_y - inner_wall_length_px // 2
        wall_y1 = wall_y0 + inner_wall_length_px

        # keep it inside the usable area and away from the border walls
        margin = self.wall_thickness_px + 1
        wall_x0 = max(start_x + margin, wall_x0)
        wall_x1 = min(end_x - margin, wall_x1)
        wall_y0 = max(start_y + margin, wall_y0)
        wall_y1 = min(end_y - margin, wall_y1)

        self.height_field_raw[wall_x0:wall_x1, wall_y0:wall_y1] = self.wall_height_px

        # =========================
        # fixed internal obstacles
        # =========================

        obstacle_height_px = self.wall_height_px

        def clamp_to_inner_area(x0, x1, y0, y1, margin=1):
            x0 = max(start_x + self.wall_thickness_px + margin, x0)
            x1 = min(end_x   - self.wall_thickness_px - margin, x1)
            y0 = max(start_y + self.wall_thickness_px + margin, y0)
            y1 = min(end_y   - self.wall_thickness_px - margin, y1)
            return x0, x1, y0, y1

        # -------------------------
        # box 1
        # size: 0.8 m x 0.6 m
        # -------------------------
        box1_w_px = max(1, int(0.8 / self.horizontal_scale))
        box1_l_px = max(1, int(0.6 / self.horizontal_scale))

        box1_center_x = start_x + int(1.2 / self.horizontal_scale)
        box1_center_y = start_y + int(1.5 / self.horizontal_scale)

        x0 = box1_center_x - box1_w_px // 2
        x1 = x0 + box1_w_px
        y0 = box1_center_y - box1_l_px // 2
        y1 = y0 + box1_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        # -------------------------
        # box 2
        # size: 1.0 m x 0.5 m
        # -------------------------
        box2_w_px = max(1, int(1.0 / self.horizontal_scale))
        box2_l_px = max(1, int(0.5 / self.horizontal_scale))

        box2_center_x = start_x + int(3.5 / self.horizontal_scale)
        box2_center_y = start_y + int(3.2 / self.horizontal_scale) + 25

        x0 = box2_center_x - box2_w_px // 2
        x1 = x0 + box2_w_px
        y0 = box2_center_y - box2_l_px // 2
        y1 = y0 + box2_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        # -------------------------
        # box 3
        # size: 1.0 m x 1.0 m
        # -------------------------
        box3_w_px = max(1, int(1.0 / self.horizontal_scale))
        box3_l_px = max(1, int(1.0 / self.horizontal_scale))

        box3_center_x = start_x + int(3.5 / self.horizontal_scale)
        box3_center_y = start_y + int(3.2 / self.horizontal_scale) + 50

        x0 = box3_center_x - box3_w_px // 2
        x1 = x0 + box3_w_px
        y0 = box3_center_y - box3_l_px // 2
        y1 = y0 + box3_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        self.heightsamples = self.height_field_raw

        # -------------------------
        # box 4
        # size: 2.0 m x 0.5 m
        # -------------------------
        box4_w_px = max(1, int(2.0 / self.horizontal_scale))
        box4_l_px = max(1, int(0.5 / self.horizontal_scale))

        box4_center_x = start_x + int(3.5 / self.horizontal_scale) + 10
        box4_center_y = start_y + int(3.2 / self.horizontal_scale) + 90

        x0 = box4_center_x - box4_w_px // 2
        x1 = x0 + box4_w_px
        y0 = box4_center_y - box4_l_px // 2
        y1 = y0 + box4_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        self.heightsamples = self.height_field_raw

        # -------------------------
        # box 5
        # size: 2.0 m x 0.5 m
        # -------------------------
        box5_w_px = max(1, int(2.0 / self.horizontal_scale))
        box5_l_px = max(1, int(0.5 / self.horizontal_scale))

        box5_center_x = start_x + int(3.5 / self.horizontal_scale) + 50
        box5_center_y = start_y + int(3.2 / self.horizontal_scale) + 20

        x0 = box5_center_x - box5_w_px // 2
        x1 = x0 + box5_w_px
        y0 = box5_center_y - box5_l_px // 2
        y1 = y0 + box5_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        self.heightsamples = self.height_field_raw

        # -------------------------
        # box 6
        # size: 0.5 m x 2.0 m
        # -------------------------
        box6_w_px = max(1, int(0.5 / self.horizontal_scale))
        box6_l_px = max(1, int(2.0 / self.horizontal_scale))

        box6_center_x = start_x + int(3.5 / self.horizontal_scale) + 61
        box6_center_y = start_y + int(3.2 / self.horizontal_scale) + 13

        x0 = box6_center_x - box6_w_px // 2
        x1 = x0 + box6_w_px
        y0 = box6_center_y - box6_l_px // 2
        y1 = y0 + box6_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        self.heightsamples = self.height_field_raw

        # -------------------------
        # box 7
        # size: 4.0 m x 0.5 m
        # -------------------------
        box7_w_px = max(1, int(4.0 / self.horizontal_scale))
        box7_l_px = max(1, int(0.5 / self.horizontal_scale))

        box7_center_x = start_x + int(3.5 / self.horizontal_scale) + 40
        box7_center_y = start_y + int(3.2 / self.horizontal_scale) + 120

        x0 = box7_center_x - box7_w_px // 2
        x1 = x0 + box7_w_px
        y0 = box7_center_y - box7_l_px // 2
        y1 = y0 + box7_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        self.heightsamples = self.height_field_raw

        # -------------------------
        # box 8
        # size: 0.5 m x 2.0 m
        # -------------------------
        box8_w_px = max(1, int(0.5 / self.horizontal_scale))
        box8_l_px = max(1, int(2.0 / self.horizontal_scale))

        box8_center_x = start_x + int(3.5 / self.horizontal_scale) + 20
        box8_center_y = start_y + int(3.2 / self.horizontal_scale) + 128

        x0 = box8_center_x - box8_w_px // 2
        x1 = x0 + box8_w_px
        y0 = box8_center_y - box8_l_px // 2
        y1 = y0 + box8_l_px
        x0, x1, y0, y1 = clamp_to_inner_area(x0, x1, y0, y1)
        self.height_field_raw[x0:x1, y0:y1] = obstacle_height_px

        self.heightsamples = self.height_field_raw
        
        self.env_origins = np.zeros((1, 1, 3), dtype=np.float32)
        self.env_origins[0, 0] = [
            self.terrain_width / 2,
            1.0,
            0.3,
        ]

        self.vertices = None
        self.triangles = None
        if self.type == "trimesh":
            self.vertices, self.triangles = terrain_utils.convert_heightfield_to_trimesh(
                self.height_field_raw,
                self.horizontal_scale,
                self.vertical_scale,
                self.slope_treshold,
            )

def generate_binary_map(height_field):
    # 1 = obstacle, 0 = free space
    binary_map = (height_field > 0).astype(np.uint8)
    return binary_map

def generate_detectable_object_positions(
    height_field,
    horizontal_scale,
    seed=0,
    num_boxes=5,
    obstacle_clearance_m=1.0,
    object_spacing_m=3.0,
):
    rng = np.random.default_rng(seed)

    obstacle_clearance_cells = int(np.ceil(obstacle_clearance_m / horizontal_scale))
    object_spacing_cells = int(np.ceil(object_spacing_m / horizontal_scale))

    obstacle_cells = np.argwhere(height_field != 0)
    placed_cells = []

    max_tries = 20000
    n_rows, n_cols = height_field.shape

    for _ in range(max_tries):
        if len(placed_cells) == num_boxes:
            break

        x_cell = int(rng.integers(0, n_rows))
        y_cell = int(rng.integers(0, n_cols))

        if height_field[x_cell, y_cell] == 1:
            continue

        valid = True

        if len(obstacle_cells) > 0:
            dx = obstacle_cells[:, 0] - x_cell
            dy = obstacle_cells[:, 1] - y_cell
            dist2 = dx * dx + dy * dy
            if np.any(dist2 < obstacle_clearance_cells * obstacle_clearance_cells):
                valid = False

        if not valid:
            continue

        for px, py in placed_cells:
            if (x_cell - px) ** 2 + (y_cell - py) ** 2 < object_spacing_cells * object_spacing_cells:
                valid = False
                break

        if not valid:
            continue

        placed_cells.append((x_cell, y_cell))

    if len(placed_cells) < num_boxes:
        raise RuntimeError(
            f"Could not place {num_boxes} objects with the requested clearances. "
            f"Placed only {len(placed_cells)}."
        )

    return [
        {"id": i, "cell_x": int(x), "cell_y": int(y)}
        for i, (x, y) in enumerate(placed_cells)
    ]


def read_robot_log_positions(log_path, horizontal_scale):
    positions_px = []

    with open(log_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            x_world = float(row["x"])
            y_world = float(row["y"])

            x_px = int(round(x_world / horizontal_scale))
            y_px = int(round(y_world / horizontal_scale))

            positions_px.append((x_px, y_px))

    return positions_px


def generate_rgb_map(height_field, object_positions=None, robot_positions=None, horizontal_scale=0.1):
    # 1 = obstacle, 0 = free space
    binary_map = (height_field > 0).astype(np.uint8)

    rgb_map = np.zeros((binary_map.shape[0], binary_map.shape[1], 3), dtype=np.uint8)
    rgb_map[binary_map == 1] = [255, 255, 255]

    # draw red circle edges around detectable objects
    if object_positions is not None:
        radius_m = 1.5
        radius_px = int(radius_m / horizontal_scale)

        obj_size_m = 0.5
        obj_half_px = int((obj_size_m / horizontal_scale) / 2)

        for obj in object_positions:
            cx = int(obj["cell_x"])
            cy = int(obj["cell_y"])

            for x in range(cx - obj_half_px, cx + obj_half_px + 1):
                for y in range(cy - obj_half_px, cy + obj_half_px + 1):
                    if 0 <= x < rgb_map.shape[0] and 0 <= y < rgb_map.shape[1]:
                        rgb_map[x, y] = [0, 0, 255]  # BLUE object

        for obj in object_positions:
            cx = int(obj["cell_x"])
            cy = int(obj["cell_y"])

            for x in range(cx - radius_px, cx + radius_px + 1):
                for y in range(cy - radius_px, cy + radius_px + 1):
                    if 0 <= x < rgb_map.shape[0] and 0 <= y < rgb_map.shape[1]:
                        dist2 = (x - cx) ** 2 + (y - cy) ** 2
                        if radius_px**2 - radius_px <= dist2 <= radius_px**2 + radius_px:
                            rgb_map[x, y] = [255, 0, 0]  # red edge

    # draw robot trajectory as green dots
    if robot_positions is not None:
        dot_radius = 1

        for cx, cy in robot_positions:
            for x in range(cx - dot_radius, cx + dot_radius + 1):
                for y in range(cy - dot_radius, cy + dot_radius + 1):
                    if 0 <= x < rgb_map.shape[0] and 0 <= y < rgb_map.shape[1]:
                        if (x - cx) ** 2 + (y - cy) ** 2 <= dot_radius ** 2:
                            rgb_map[x, y] = [0, 255, 0]  # green dot

    return rgb_map


if __name__ == "__main__":
    terrain = Terrain()

    height_field = terrain.height_field_raw

    binary_map = generate_binary_map(height_field)

    # visualize
    plt.imshow(binary_map, cmap="gray")
    plt.title("Binary Occupancy Map")
    plt.show()

    seed = 0
    log_path = "robot_log.csv"

    object_positions = generate_detectable_object_positions(
        height_field=height_field,
        horizontal_scale=terrain.horizontal_scale,
        seed=seed,
        num_boxes=5,
        obstacle_clearance_m=1.0,
        object_spacing_m=3.0,
    )

    print("[Detectable object positions]")
    for obj in object_positions:
        print(f"object {obj['id']}: cell=({obj['cell_x']}, {obj['cell_y']})")

    robot_positions = read_robot_log_positions(
        log_path=log_path,
        horizontal_scale=terrain.horizontal_scale
    )

    print(f"[Robot trajectory] loaded {len(robot_positions)} positions from {log_path}")

    rgb_map = generate_rgb_map(
        height_field,
        object_positions=object_positions,
        robot_positions=robot_positions,
        horizontal_scale=terrain.horizontal_scale
    )

    plt.imshow(rgb_map)
    plt.title("Occupancy Map")
    plt.show()
