# Запуск контроллера через ROS 2

## 1. Запуск bridge-ноды (Терминал 1)

Сначала запустите bridge-ноду для установления соединения между контроллером и симулятором через TCP/UDP:

```bash
cd ~/ros2_isaac_bridge/ros2_ws
ros2 run ros2_bridge_pkg bridge_node
```

---

## 2. Запуск симулятора (Терминал 2)

Во втором терминале запустите симулятор:

```bash
cd ~/ros2_isaac_bridge/sim_side
python isaac_controller.py
```

После этого симулятор будет успешно подключён к bridge-ноду и готов публиковать данные и принимать команды скорости.

---

## 3. Запуск контроллера (Терминал 3)

В третьем терминале запустите прототип контроллера после добавления вашего решения:

```bash
cd ~/ros2_isaac_bridge/sim_side
python3 controller.py
```

---

## ⚠️ Важные замечания

- Изменять **основную логику (runner)** в `isaac_controller.py` **ЗАПРЕЩЕНО**.  
- Разрешается изменять **значение seed** для тестирования алгоритма в разных сценариях.

---

## Передача информации об обнаружении объектов (ОБЯЗАТЕЛЬНО)

Вы обязаны отправлять сообщение из вашего контроллера в `isaac_controller.py` при обнаружении объекта.

### Пример (publisher в `controller.py`):

```python
self.detected_object_pub = self.create_publisher(std_msgs.msg.Int32, "/competition/detected_object", 10)

def publish_detected_object(self, object_id: int):
    msg = Int32()
    msg.data = int(object_id)
    self.detected_object_pub.publish(msg)
```

---

### Пример (subscriber в `isaac_controller.py`):

Создайте отдельную ROS 2 ноду:

```python
class DetectedObjectListener(Node):
    def __init__(self):
        super().__init__("isaac_listener")

        self.latest_detected = None

        self.sub = self.create_subscription(
            Int32,
            "/competition/detected_object",
            self.callback,
            10
        )

    def callback(self, msg):
        self.latest_detected = int(msg.data)
```

Инициализируйте ноду внутри `main()`:

```python
rclpy.init()
ros_node = DetectedObjectListener()
```

---

## ⚠️ Обязательное требование

Вы **обязаны заменить строки 240–242** в `isaac_controller.py` на проверку, основанную на подписке (subscription), используя созданную ROS-ноду.

---

## ⚠️ Ограничения

- **ЗАПРЕЩЕНО** добавлять logging-топики в `BridgeNode`.  
- Передача информации для логирования должна выполняться **через отдельную ROS 2 ноду** между контроллером и `isaac_controller.py`.
