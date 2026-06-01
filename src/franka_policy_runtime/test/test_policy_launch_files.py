from pathlib import Path


_LAUNCH_DIR = Path(__file__).parents[1] / "launch"
EYE_TO_HAND_CAMERA = "eye_to_hand_camera"


def test_camera_launches_publish_camera_link_to_optical_tf():
    for launch_name in ["bc_cube_stack.launch.py", "vla_policy.launch.py"]:
        source = (_LAUNCH_DIR / launch_name).read_text(encoding="utf-8")

        assert 'executable="static_transform_publisher"' not in source
        assert '"child_frame": f"{EYE_TO_HAND_CAMERA}_{EYE_TO_HAND_CAMERA}_link"' in source
        assert 'f"{EYE_TO_HAND_CAMERA}_color_optical_frame"' in source
