import math
from pathlib import Path

import yaml

import pytest

from franka_telep.franka_mapping import (
    FR3_READY_JOINTS,
    map_gripper_offset_to_width,
    map_servo_offsets_to_joints,
)
from franka_telep.zhonglin_protocol import pwm_to_angle, smooth_toward


def test_pwm_to_angle_parses_zhonglin_response():
    assert pwm_to_angle("#003P1500!", 3) == pytest.approx(135.0)


def test_pwm_to_angle_returns_none_for_wrong_servo():
    assert pwm_to_angle("#004P1500!", 3) is None


def test_smooth_toward_uses_alpha():
    assert smooth_toward([0.0, 10.0], [10.0, 0.0], alpha=0.2) == pytest.approx([2.0, 8.0])


def test_map_servo_offsets_to_joints_applies_degrees_signs_scales_and_limits():
    result = map_servo_offsets_to_joints(
        [90.0, 90.0],
        base_positions_rad=[0.0, 0.0],
        arm_servo_indices=[0, 1],
        signs=[1.0, -1.0],
        scales=[1.0, 0.5],
        lower_limits=[-2.0, -2.0],
        upper_limits=[2.0, 2.0],
    )
    assert result == pytest.approx([math.pi / 2.0, -math.pi / 4.0])


def test_map_servo_offsets_to_joints_clamps_limits():
    result = map_servo_offsets_to_joints(
        [180.0],
        base_positions_rad=[0.0],
        arm_servo_indices=[0],
        signs=[1.0],
        scales=[1.0],
        lower_limits=[-1.0],
        upper_limits=[1.0],
    )
    assert result == pytest.approx([1.0])


def test_map_gripper_offset_to_width():
    assert map_gripper_offset_to_width(
        15.0,
        open_offset_deg=0.0,
        closed_offset_deg=30.0,
        min_width=0.0,
        max_width=0.08,
    ) == pytest.approx(0.04)


def test_map_gripper_offset_to_width_clamps_out_of_range():
    assert map_gripper_offset_to_width(
        -10.0,
        open_offset_deg=0.0,
        closed_offset_deg=30.0,
        min_width=0.0,
        max_width=0.08,
    ) == pytest.approx(0.08)
    assert map_gripper_offset_to_width(
        60.0,
        open_offset_deg=0.0,
        closed_offset_deg=30.0,
        min_width=0.0,
        max_width=0.08,
    ) == pytest.approx(0.0)


def test_real_teleop_config_uses_absolute_home_and_home_gate():
    config_path = Path(__file__).parents[1] / "config" / "franka_telep.yaml"
    config = yaml.safe_load(config_path.read_text())
    teleop = config["franka_teleop"]["ros__parameters"]
    home = config["franka_home_initializer"]["ros__parameters"]

    assert teleop["use_absolute_servo_angles"] is True
    assert teleop["latch_uarm_home_on_ready"] is True
    assert teleop["require_home_ready"] is True
    assert teleop["use_current_as_initial"] is False
    assert teleop["max_joint_step_rad"] >= 0.0
    assert teleop["command_rate_hz"] >= 20.0
    assert teleop["trajectory_duration_sec"] <= 0.3
    assert 0.0 < teleop["target_filter_alpha"] < 1.0
    assert 5.0 <= teleop["cd_omega"] <= 100.0
    assert 0.0 < teleop["servo_filter_alpha"] <= 1.0
    assert teleop["target_deadband_rad"] > 0.0
    assert teleop["joint_state_topic"] == "/franka/joint_states"
    assert teleop["initial_joint_positions"] == home["home_joint_positions"]
    assert teleop["initial_joint_positions"] == FR3_READY_JOINTS
    assert home["home_joint_positions"][3] == pytest.approx(-2.95)
    assert home["home_joint_positions"][3] >= teleop["joint_lower_limits"][3] + teleop["joint_limit_margin_rad"]
    assert len(teleop["uarm_home_abs_angles_deg"]) == 8


def test_servo_reader_config_uses_low_latency_polling():
    config_path = Path(__file__).parents[1] / "config" / "franka_telep.yaml"
    config = yaml.safe_load(config_path.read_text())
    reader = config["zhonglin_servo_reader"]["ros__parameters"]

    assert reader["publish_rate_hz"] >= 120.0
    assert reader["command_delay_sec"] <= 0.006
    assert reader["read_retries"] >= 3
    assert reader["smoothing_alpha"] >= 0.6


def test_uarm_leader_config_drives_franka_gripper():
    config_path = Path(__file__).parents[1] / "config" / "franka_telep.yaml"
    config = yaml.safe_load(config_path.read_text())
    leader = config["uarm_leader_publisher"]["ros__parameters"]

    assert leader["enable_gripper"] is True
    assert leader["gripper_move_action"] == "/franka_gripper/move"
    assert leader["gripper_grasp_action"] == "/franka_gripper/grasp"
    assert leader["gripper_servo_index"] == 7
    assert leader["gripper_threshold_deg"] > leader["gripper_hysteresis_deg"]
    assert len(leader["max_joint_velocity_rad_s"]) == 7
    assert max(leader["max_joint_velocity_rad_s"]) <= 0.5
