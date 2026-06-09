from pathlib import Path


_LAUNCH_DIR = Path(__file__).parents[1] / "launch"
EYE_TO_HAND_CAMERA = "eye_to_hand_camera"


def test_camera_launches_publish_camera_link_to_optical_tf():
    for launch_name in ["bc_cube_stack.launch.py", "vla_policy.launch.py"]:
        source = (_LAUNCH_DIR / launch_name).read_text(encoding="utf-8")

        assert 'executable="static_transform_publisher"' not in source
        assert '"child_frame": f"{EYE_TO_HAND_CAMERA}_{EYE_TO_HAND_CAMERA}_link"' in source
        assert 'f"{EYE_TO_HAND_CAMERA}_color_optical_frame"' in source


def test_continuous_policy_launches_expose_controller_mode():
    for launch_name in ["bc_cube_stack.launch.py", "vla_policy.launch.py"]:
        source = (_LAUNCH_DIR / launch_name).read_text(encoding="utf-8")
        assert 'DeclareLaunchArgument("control_mode", default_value="cartesian_delta"' in source
        assert '"controller_mode": LaunchConfiguration("control_mode")' in source
        assert '"control_mode": LaunchConfiguration("control_mode")' in source


def test_absolute_pose_launches_keep_trajectory_controller():
    for launch_name in ["action_test.launch.py", "anygrasp.launch.py"]:
        source = (_LAUNCH_DIR / launch_name).read_text(encoding="utf-8")
        assert '"controller_mode": "trajectory"' in source
