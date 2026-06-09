# Franka Teleop OpenVLA Dataset

Build and start the existing uArm follower teleoperation together with the
eye-to-hand camera and recorder:

```bash
cd /home/young/ros2_ws
colcon build --packages-select franka_telep --symlink-install
source install/setup.bash
ros2 launch franka_telep uarm_openvla_collect.launch.py \
  robot_ip:=172.16.0.2 \
  instruction:="pick up the red block"
```

Recording starts after `/franka_teleop/home_ready` becomes true. Stop and save
the current episode with:

```bash
ros2 topic pub --once /franka_teleop/dataset_recording \
  std_msgs/msg/Bool "{data: false}"
```

Set the instruction for the next episode, then start recording again:

```bash
ros2 topic pub --once /franka_teleop/dataset_instruction \
  std_msgs/msg/String "{data: 'place the red block in the bowl'}"
ros2 topic pub --once /franka_teleop/dataset_recording \
  std_msgs/msg/Bool "{data: true}"
```

The recorder writes raw episodes under:

```text
<dataset_root>/<dataset_name>/raw/episode_XXXXXX/
```

Each episode contains JPEG images, `steps.npz`, and `episode.json`. Convert
them to TFDS/RLDS in an environment with TensorFlow and TensorFlow Datasets:

```bash
export FRANKA_TELEOP_RAW_DIR=~/franka_openvla_data/franka_teleop/raw
cd openvla_rlds/franka_teleop_dataset
tfds build --overwrite --data_dir ~/tensorflow_datasets
```

The resulting step schema is:

```text
observation.image: uint8 [256, 256, 3]
observation.wrist_image: uint8 [256, 256, 3]
observation.state: float32 [8]  # XYZ, RPY, padding, gripper
observation.joint_positions: float32 [7]
action: float32 [7]             # delta XYZ, delta RPY, gripper
language_instruction: string
is_first/is_last/is_terminal, reward, discount
```

The gripper channel follows the Bridge/OpenVLA convention: `0.0` is closed
and `1.0` is open. Actions are relabeled from each observation to the next
observed TCP pose, rather than copied from the uArm command.

Set `wrist_image_topic` when a wrist camera is available. For a single-camera
episode, the TFDS builder fills `wrist_image` with the primary image so the
schema remains stable.

Use `openvla_registration.py` as the registration template in an OpenVLA
checkout. The dataset already uses `StateEncoding.POS_EULER` and
`ActionEncoding.EEF_POS`.
