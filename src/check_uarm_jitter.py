#!/usr/bin/env python3
"""Check UArm servo signal jitter. Use while arm is stationary."""

import time
import statistics
import argparse

from franka_telep.zhonglin_protocol import ZhonglinServoBus


def main():
    parser = argparse.ArgumentParser(description="Check UArm servo jitter")
    parser.add_argument("--port", default="/dev/ttyUSB0")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--threshold", type=float, default=5.0, help="pp threshold for ✗ flag (°)")
    parser.add_argument("--ids", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 6])
    args = parser.parse_args()

    bus = ZhonglinServoBus(args.port, 115200, timeout_sec=0.04, command_delay_sec=0.006)
    bus.probe()

    samples = {sid: [] for sid in args.ids}
    print(f"采样 {args.duration}s，请勿移动 UArm...")
    t0 = time.monotonic()
    while time.monotonic() - t0 < args.duration:
        for sid in args.ids:
            angle, _ = bus.read_angle(sid, pwm_min=500, pwm_max=2500, angle_range_deg=270.0)
            if angle is not None:
                samples[sid].append(angle)
    bus.close()

    print(f"\n{'Servo':>6s}  {'样本':>5s}  {'均值(°)':>9s}  {'std(°)':>8s}  {'pp(°)':>8s}")
    print("-" * 44)
    for sid in args.ids:
        data = samples[sid]
        if len(data) < 2:
            print(f"{sid:>6d}  {len(data):>5d}  数据不足")
            continue
        mean_v = statistics.mean(data)
        std_v = statistics.stdev(data)
        pp = max(data) - min(data)
        flag = " ✓" if pp <= 2 else (" △" if pp <= args.threshold else " ✗")
        print(f"{sid:>6d}  {len(data):>5d}  {mean_v:>9.3f}  {std_v:>8.4f}  {pp:>8.4f}{flag}")

    print(f"\n✓ <2°  △ 2-5°  ✗ >{args.threshold}°")


if __name__ == "__main__":
    main()
