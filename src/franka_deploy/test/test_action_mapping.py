import numpy as np
import unittest

from franka_deploy.action_mapping import TwistLimits, action_to_twist, gripper_should_close, validate_action


class ActionMappingTest(unittest.TestCase):
    def test_validate_action_requires_seven_values(self):
        with self.assertRaises(ValueError):
            validate_action([0.0] * 6)

    def test_validate_action_rejects_nan(self):
        with self.assertRaises(ValueError):
            validate_action([0.0, 0.0, np.nan, 0.0, 0.0, 0.0, 0.0])

    def test_action_to_twist_scales_and_clips_delta_motion(self):
        limits = TwistLimits(
            max_linear_velocity=0.05,
            max_angular_velocity=0.25,
            max_linear_step=0.01,
            max_angular_step=0.05,
        )

        dt = 0.2  # 1 / 5 Hz
        linear, angular = action_to_twist(
            [0.2, -0.2, 0.005, 1.0, -1.0, 0.01, 0.0], dt, limits,
        )

        np.testing.assert_array_almost_equal(linear, [0.05, -0.05, 0.025])
        np.testing.assert_array_almost_equal(angular, [0.25, -0.25, 0.05])

    def test_gripper_threshold_uses_last_action_dimension(self):
        self.assertTrue(gripper_should_close([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5], 0.5))
        self.assertFalse(gripper_should_close([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.49], 0.5))


if __name__ == '__main__':
    unittest.main()
