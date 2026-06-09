"""Registration snippets for the OpenVLA checkout in this workspace.

Add the config values to OXE_DATASET_CONFIGS and register the identity
standardization transform under the ``franka_teleop_dataset`` TFDS name.
"""

from prismatic.vla.datasets.rlds.oxe.configs import ActionEncoding, StateEncoding


FRANKA_TELEOP_OXE_CONFIG = {
    "image_obs_keys": {
        "primary": "image",
        "secondary": None,
        "wrist": "wrist_image",
    },
    "depth_obs_keys": {"primary": None, "secondary": None, "wrist": None},
    "state_obs_keys": ["state"],
    "state_encoding": StateEncoding.POS_EULER,
    "action_encoding": ActionEncoding.EEF_POS,
}


def franka_teleop_dataset_transform(trajectory):
    """The recorder already emits standardized EEF delta actions."""
    return trajectory
