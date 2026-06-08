import re
import time
from collections.abc import Sequence


def pwm_to_angle(
    response: str,
    servo_id: int,
    *,
    pwm_min: int = 500,
    pwm_max: int = 2500,
    angle_range_deg: float = 270.0,
) -> float | None:
    match = re.search(rf"#{servo_id:03d}P(\d{{4}})", response)
    if match is None:
        return None
    pwm_value = int(match.group(1))
    return (pwm_value - pwm_min) / (pwm_max - pwm_min) * angle_range_deg


def smooth_toward(
    current: Sequence[float],
    target: Sequence[float],
    *,
    alpha: float,
) -> list[float]:
    return [float(value + (goal - value) * alpha) for value, goal in zip(current, target)]


class ZhonglinServoBus:
    def __init__(
        self,
        port: str,
        baudrate: int,
        *,
        timeout_sec: float = 0.1,
        command_delay_sec: float = 0.008,
        exclusive: bool = True,
    ):
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("pyserial is required to use the Zhonglin servo reader") from exc

        try:
            self._serial = serial.Serial(port, baudrate, timeout=timeout_sec, exclusive=exclusive)
        except TypeError:
            self._serial = serial.Serial(port, baudrate, timeout=timeout_sec)
        self._command_delay_sec = command_delay_sec

    def close(self) -> None:
        self._serial.close()

    def send_command(self, command: str) -> str:
        self._serial.reset_input_buffer()
        self._serial.write(command.encode("ascii"))
        self._serial.flush()
        time.sleep(self._command_delay_sec)
        return self._serial.read_all().decode("ascii", errors="ignore")

    def read_angle(
        self,
        servo_id: int,
        *,
        pwm_min: int = 500,
        pwm_max: int = 2500,
        angle_range_deg: float = 270.0,
    ) -> tuple[float | None, str]:
        response = self.send_command(f"#{servo_id:03d}PRAD!")
        angle = pwm_to_angle(
            response.strip(),
            servo_id,
            pwm_min=pwm_min,
            pwm_max=pwm_max,
            angle_range_deg=angle_range_deg,
        )
        return angle, response

    def unlock_servo(self, servo_id: int) -> None:
        self.send_command("#000PCSK!")
        self.send_command(f"#{servo_id:03d}PULK!")

    def probe(self) -> None:
        self.send_command("#000PVER!")
