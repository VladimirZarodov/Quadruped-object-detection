# Aliengo Competition

Репозиторий для соревнования по управлению Aliengo в Isaac Gym.

Проект поддерживает три рабочих сценария:

1. Python-контроллер в Docker для быстрой проверки логики.
2. Полный ROS 2 режим в Docker для работы через топики и bridge.
3. Локальный Isaac Gym на хосте и ROS 2 bridge в контейнере, если нужно снизить расход GPU-памяти по сравнению с Docker-симуляцией.

## Быстрый выбор режима

- Если нужно просто проверить свою логику: используйте `scripts/controller.py`.
- Если нужно работать через ROS-топики: используйте `ros2_isaac_bridge/sim_side/isaac_controller.py` вместе с ROS 2 bridge.
- Если в Docker Isaac Gym потребляет слишком много VRAM: запускайте Isaac Gym локально на хосте, а ROS 2 контейнер оставляйте как есть.

## Где писать свою логику

Основная пользовательская логика находится в `src/aliengo_competition/controllers/main_controller.py`.

Внутри файла есть два ключевых блока:

- `USER PARAMETERS START / END` - параметры поведения.
- `USER CONTROL LOGIC START / END` - код, который формирует команды роботу.

Внутри `RobotState` доступны, в частности:

- `state.q`, `state.q_dot` - положения и скорости суставов
- `state.joints.name` - имена суставов
- `state.vx`, `state.vy`, `state.wz` - линейная и угловая скорость корпуса
- `state.base_linear_velocity_xyz` - полный вектор линейной скорости
- `state.base_angular_velocity_xyz` - полный вектор угловой скорости
- `state.imu.angular_velocity_xyz` - данные IMU
- `state.camera.rgb`, `state.camera.depth` - данные камер
- `object_queue` - очередь объектов

Для логирования найденного объекта нужно сохранить шаблон:

- `get_found_object_id(...)` - логика определения объекта
- `log_found_object(...)` - запись результата в судейский лог

## Структура проекта

- `src/aliengo_competition/controllers/main_controller.py` - основная логика участника
- `scripts/controller.py` - точка входа для Python-режима
- `scripts/play.py` - локальный запуск/проверка окружения Isaac Gym
- `ros2_isaac_bridge/sim_side/isaac_controller.py` - симуляция в ROS-сценарии
- `ros2_isaac_bridge/sim_side/sim_bridge_client.py` - сокетный клиент между симуляцией и ROS bridge
- `ros2_isaac_bridge/ros2_ws/src/ros2_bridge_pkg/ros2_bridge_pkg/bridge_node.py` - ROS 2 bridge node
- `docker/` - Dockerfile, compose-файлы и утилиты запуска
- `docker/ctl.sh` - основной скрипт для сборки и запуска контейнеров

## Архитектура ROS 2 режима

ROS-сценарий разделён на две части:

- `aliengo-competition` - Isaac Gym и симуляция
- `ros2-jazzy` - ROS 2 Jazzy, bridge node, `rqt`, `rviz2` и отладочные инструменты

Схема обмена:

```text
ROS 2 node /cmd_vel
        |
        v
bridge_node.py  <---->  SimBridgeClient
        ^                      ^
        |                      |
        |                  isaac_controller.py
        |
   ROS topics
```

Используемые порты:

- `5005` - команды `vx`, `vy`, `wz` из ROS в симуляцию
- `5006` - скорость корпуса
- `5007` - RGB изображение
- `5008` - depth изображение
- `5009` - суставы
- `5010` - IMU

Bridge публикует топики:

- `/aliengo/base_velocity`
- `/aliengo/camera/color/image_raw`
- `/aliengo/camera/depth/image_raw`
- `/aliengo/joint_states`
- `/aliengo/imu`

Bridge принимает:

- `/cmd_vel`

ROS domain ID для работы в локальной сети:

- В консоли *вне* контейнеров: `docker network create ros_net_NUMBER` - где вместо NUMBER номер команды (или любое другое обозначение, на ваше усмотрение). Выполняется на каждом компьютере до запуска контейнеров.
- При запуске контенйнера *run* добавить флаг `--network ros_net_NUMBER`
- Внутри контейнера, перед стартом ROS2 нод `export ROS_DOMAIN_ID=**`, где вместо ** будет выданное вам нами двузначное число. Выполняется на каждом компьютере до первого старта ноды внутри контейнера в его консоли. В случае перезапуска - прописать заново.

## Подготовка окружения

### Требования для Docker-варианта

Нужны:

- Docker
- Docker Compose
- NVIDIA драйверы
- `nvidia-container-toolkit`
- X11 и `xhost`, если нужна визуализация

Если вы запускаете GUI из контейнера, переменная `DISPLAY` должна быть задана на хосте. Скрипт `docker/ctl.sh` сам настраивает доступ к X-серверу, если окружение уже готово.

### Требования для локального Isaac Gym

Нужны:

- Conda
- Python `3.8`
- локальная копия Isaac Gym внутри `docker/isaac-gym/isaacgym`

## Сценарий 1. Python-контроллер в Docker

Этот режим подходит для простой проверки логики без ROS.

### 1. Поднять контейнер симуляции

```bash
docker/ctl.sh up
```

### 2. Открыть shell в контейнере

```bash
docker/ctl.sh exec
```

### 3. Запустить контроллер

```bash
python scripts/controller.py --steps 15000 --seed 0
```

Полезные флаги:

- `--no_render_camera` - отключить окно камеры
- `--steps` - ограничить число шагов
- `--seed` - зафиксировать seed

## Сценарий 2. Полный ROS 2 режим в Docker

Этот режим нужен, если вы хотите работать через ROS 2 топики и стандартный bridge.

### 1. Поднять контейнер симуляции

```bash
docker/ctl.sh up
```

### 2. Собрать ROS 2 слой

```bash
docker/ctl.sh ros2-build
```

### 3. Поднять ROS 2 контейнер

```bash
docker/ctl.sh ros2-up
```

### 4. Запустить симуляцию

В отдельном терминале:

```bash
docker/ctl.sh exec
python ros2_isaac_bridge/sim_side/isaac_controller.py
```

### 5. Запустить ROS bridge

В другом терминале:

```bash
docker/ctl.sh ros2-exec
bash /workspace/aliengo_competition/ros2_isaac_bridge/run_bridge_node.sh
```

### 6. Использовать ROS-инструменты

Дополнительные терминалы:

```bash
docker/ctl.sh exec
```

или:

```bash
docker/ctl.sh ros2-exec
```

Полезно для:

- `ros2 topic list`
- `ros2 topic echo /aliengo/base_velocity`
- `rqt_graph`
- `rviz2`

## Сценарий 3. Локальный Isaac Gym на хосте + ROS 2 bridge в контейнере

Этот вариант нужен, если симуляция внутри Docker потребляет слишком много GPU-памяти, а при локальном запуске на хосте работает заметно легче.

В этой схеме:

- Isaac Gym запускается локально на хосте
- ROS 2 bridge остаётся в контейнере
- контейнер `aliengo-competition` для симуляции не нужен

Это работает, потому что `bridge_node.py` и `SimBridgeClient` общаются через `host network` и сокеты на портах `5005-5010`.

### 1. Создать conda environment

```bash
conda create -y -n roboschool python=3.8
conda activate roboschool
```

### 2. Установить проект

```bash
cd "$HOME/workspace/roboschool_competition"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

### 3. Установить Isaac Gym Python package

```bash
cd "$HOME/workspace/roboschool_competition/docker/isaac-gym/isaacgym/python"
python -m pip install -e .
```

### 4. Один раз прописать `PYTHONPATH`

Этот шаг делается один раз, например через `~/.bashrc`:

```bash
echo 'export PYTHONPATH="$HOME/workspace/roboschool_competition/docker/isaac-gym/isaacgym/python:$HOME/workspace/roboschool_competition/src:$HOME/workspace/roboschool_competition:${PYTHONPATH}"' >> ~/.bashrc
source ~/.bashrc
```
```bash
conda activate roboschool
```
### 5. В каждом новом shell после `conda activate` экспортировать `LD_LIBRARY_PATH`

Этот шаг нужно выполнять каждый раз после активации окружения:

```bash
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
```

### 6. Проверить, что окружение собрано

```bash
python -c "import isaacgym, torch, aliengo_gym; print('ok')"
```

### 7. Поднять только ROS 2 контейнер

```bash
cd "$HOME/workspace/roboschool_competition"
docker/ctl.sh ros2-build
docker/ctl.sh ros2-up
```

### 8. Сначала запустить bridge в контейнере

```bash
cd "$HOME/workspace/roboschool_competition"
docker/ctl.sh ros2-exec
bash /workspace/aliengo_competition/ros2_isaac_bridge/run_bridge_node.sh
```

Bridge лучше запускать раньше симуляции, потому что depth-канал использует TCP на порту `5008`, и `isaac_controller.py` ожидает, что этот порт уже слушается.

### 9. Запустить Isaac Gym локально на хосте

Для Python-варианта:

```bash
cd "$HOME/workspace/roboschool_competition"
conda activate roboschool
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
python scripts/controller.py --steps 15000 --seed 0
```

Для ROS-варианта:

```bash
cd "$HOME/workspace/roboschool_competition"
conda activate roboschool
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CONDA_PREFIX}/lib"
python ros2_isaac_bridge/sim_side/isaac_controller.py
```

## Как контейнеры и процессы общаются

Обмен построен не как прямое взаимодействие обычных ROS-нод между контейнерами, а через bridge и сокеты.

Последовательность такая:

1. ROS 2 нода публикует `cmd_vel`.
2. `bridge_node.py` принимает команду.
3. `SimBridgeClient` передаёт её в симуляцию через порт `5005`.
4. Симуляция отправляет обратно скорость, RGB, depth, IMU и суставы.
5. Bridge публикует это как ROS 2 топики.

Из-за `network_mode: host` ROS 2 контейнер может работать как с симуляцией в Docker, так и с `isaac_controller.py`, запущенным прямо на хосте.

## Как устроены режимы запуска

### Python-вариант

Основные файлы:

- `scripts/controller.py`
- `src/aliengo_competition/controllers/main_controller.py`

Что происходит:

- `scripts/controller.py` разбирает аргументы
- создаёт интерфейс робота
- вызывает основной цикл управления

### ROS-вариант

Основные файлы:

- `ros2_isaac_bridge/sim_side/isaac_controller.py`
- `ros2_isaac_bridge/sim_side/sim_bridge_client.py`
- `ros2_isaac_bridge/ros2_ws/src/ros2_bridge_pkg/ros2_bridge_pkg/bridge_node.py`

Что происходит:

- `isaac_controller.py` шагает симуляцию и собирает телеметрию
- `sim_bridge_client.py` передаёт команды и данные через сокеты
- `bridge_node.py` публикует их в ROS 2 и слушает `/cmd_vel`

## Полезные команды

```bash
docker/ctl.sh build
docker/ctl.sh up
docker/ctl.sh exec
docker/ctl.sh down

docker/ctl.sh ros2-build
docker/ctl.sh ros2-up
docker/ctl.sh ros2-exec
docker/ctl.sh ros2-down
```

## Работа в своей копии репозитория

Если вы ведёте разработку в своём fork:

```bash
git clone git@github.com:<your-user>/roboschool_competition.git
cd roboschool_competition
git remote add upstream git@github.com:<original-owner>/roboschool_competition.git
git fetch upstream
git checkout -b my-competition-solution
git push -u origin my-competition-solution
```

Рекомендуется:

- работать в отдельной ветке
- не коммитить напрямую в `main`
- периодически подтягивать изменения из `upstream`
