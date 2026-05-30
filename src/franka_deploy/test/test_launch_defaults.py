from pathlib import Path
import unittest


class LaunchDefaultsTest(unittest.TestCase):
    def test_openvla_launch_defaults_are_safe(self):
        package_root = Path(__file__).resolve().parents[1]
        source = (package_root / 'launch' / 'openvla_franka.launch.py').read_text()

        self.assertIn("DeclareLaunchArgument('load_in_4bit', default_value='true')", source)
        self.assertIn("DeclareLaunchArgument('use_fake_hardware', default_value='false')", source)
        self.assertNotIn("enable_on_start", source)
        self.assertIn("DeclareLaunchArgument('openvla_path', default_value='openvla/openvla-7b')", source)


if __name__ == '__main__':
    unittest.main()
