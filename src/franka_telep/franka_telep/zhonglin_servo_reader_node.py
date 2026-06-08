from __future__ import annotations

import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from franka_telep.zhonglin_protocol import ZhonglinServoBus, pwm_to_angle, smooth_toward


class ZhonglinServoReaderNode(Node):
    def __init__(self) -> None:
        super().__init__("zhonglin_servo_reader")
        self.declare_parameter("port", "/dev/ttyUSB0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("servo_ids", [0, 1, 2, 3, 4, 5, 6])
        self.declare_parameter("publish_topic", "/servo_angles")
        self.declare_parameter("absolute_publish_topic", "/servo_absolute_angles")
        self.declare_parameter("publish_rate_hz", 120.0)
        self.declare_parameter("serial_timeout_sec", 0.04)
        self.declare_parameter("command_delay_sec", 0.006)
        self.declare_parameter("exclusive_serial", True)
        self.declare_parameter("read_retries", 3)
        self.declare_parameter("pwm_min", 500)
        self.declare_parameter("pwm_max", 2500)
        self.declare_parameter("angle_range_deg", 270.0)
        self.declare_parameter("step_threshold_deg", 0.2)
        self.declare_parameter("smoothing_alpha", 1.0)
        self.declare_parameter("danger_jump_deg", 90.0)
        self.declare_parameter("calibrate_on_start", True)

        self._servo_ids = [int(value) for value in self.get_parameter("servo_ids").value]
        if not self._servo_ids:
            raise ValueError("servo_ids must not be empty")

        self._pwm_min = int(self.get_parameter("pwm_min").value)
        self._pwm_max = int(self.get_parameter("pwm_max").value)
        self._angle_range_deg = float(self.get_parameter("angle_range_deg").value)
        self._step_threshold_deg = float(self.get_parameter("step_threshold_deg").value)
        self._smoothing_alpha = float(self.get_parameter("smoothing_alpha").value)
        self._danger_jump_deg = float(self.get_parameter("danger_jump_deg").value)

        self._pub = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("publish_topic").value),
            10,
        )
        self._absolute_pub = self.create_publisher(
            Float64MultiArray,
            str(self.get_parameter("absolute_publish_topic").value),
            10,
        )
        self._bus = ZhonglinServoBus(
            str(self.get_parameter("port").value),
            int(self.get_parameter("baudrate").value),
            timeout_sec=float(self.get_parameter("serial_timeout_sec").value),
            command_delay_sec=float(self.get_parameter("command_delay_sec").value),
            exclusive=bool(self.get_parameter("exclusive_serial").value),
        )
        self._zero_angles = [0.0] * len(self._servo_ids)
        self._absolute_angles = [0.0] * len(self._servo_ids)
        self._target_offsets = [0.0] * len(self._servo_ids)
        self._offsets = [0.0] * len(self._servo_ids)

        self._data_lock = threading.Lock()
        self._reader_running = threading.Event()
        self._reader_thread: threading.Thread | None = None

        if bool(self.get_parameter("calibrate_on_start").value):
            self._calibrate()

        self._reader_running.set()
        self._reader_thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader_thread.start()

        period = 1.0 / max(1.0, float(self.get_parameter("publish_rate_hz").value))
        self._timer = self.create_timer(period, self._tick)
        self.get_logger().info(
            f"Zhonglin servo reader publishing {len(self._servo_ids)} channels to "
            f"{self.get_parameter('publish_topic').value}"
        )

    def destroy_node(self) -> bool:
        self._reader_running.clear()
        if self._reader_thread is not None:
            self._reader_thread.join(timeout=2.0)
        try:
            self._bus.close()
        finally:
            return super().destroy_node()

    def _calibrate(self) -> None:
        self.get_logger().info("Calibrating servo zero angles. Do not move the teleop arm.")
        self._bus.probe()
        for index, servo_id in enumerate(self._servo_ids):
            self._bus.unlock_servo(servo_id)
            angle, response = self._read_angle_with_retries(servo_id)
            if angle is None:
                raise RuntimeError(
                    f"Servo {servo_id} calibration failed after retries; last response: {response.strip()}"
                )
            self._zero_angles[index] = angle
        self.get_logger().info(f"Servo zero angles: {[round(value, 2) for value in self._zero_angles]}")

    def _reader_loop(self) -> None:
        """Continuously read all servos in a background thread.

        Updates ``_target_offsets`` and ``_absolute_angles`` under the data lock
        so the ROS timer callback can publish the latest values without blocking.
        """
        while self._reader_running.is_set():
            for index, servo_id in enumerate(self._servo_ids):
                if not self._reader_running.is_set():
                    break
                angle, response = self._read_angle_with_retries(servo_id)
                if angle is None:
                    self.get_logger().warn(
                        f"Servo {servo_id} response error: {response.strip()}",
                        throttle_duration_sec=1.0,
                    )
                    continue

                new_offset = angle - self._zero_angles[index]
                with self._data_lock:
                    self._absolute_angles[index] = angle
                    current_target = self._target_offsets[index]

                if abs(new_offset - current_target) > self._danger_jump_deg:
                    self.get_logger().error(
                        f"Servo {servo_id} angle jump too large: "
                        f"{new_offset:.2f} vs {current_target:.2f} deg"
                    )
                    continue
                if abs(new_offset - current_target) > self._step_threshold_deg:
                    with self._data_lock:
                        self._target_offsets[index] = new_offset

    def _tick(self) -> None:
        """Publish latest servo offsets (ROS timer callback, non-blocking)."""
        with self._data_lock:
            targets = list(self._target_offsets)
            absolutes = list(self._absolute_angles)

        self._offsets = smooth_toward(self._offsets, targets, alpha=self._smoothing_alpha)
        self._pub.publish(Float64MultiArray(data=self._offsets))
        self._absolute_pub.publish(Float64MultiArray(data=absolutes))

    def _read_angle_with_retries(self, servo_id: int) -> tuple[float | None, str]:
        last_response = ""
        for _ in range(max(1, int(self.get_parameter("read_retries").value))):
            angle, response = self._bus.read_angle(
                servo_id,
                pwm_min=self._pwm_min,
                pwm_max=self._pwm_max,
                angle_range_deg=self._angle_range_deg,
            )
            last_response = response
            if angle is not None:
                return angle, response
        return None, last_response


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ZhonglinServoReaderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
