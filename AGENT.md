# FR3 工作区指南

ROS 2 Humble colcon 工作区，Franka FR3 机械臂 + RealSense D435I。

## 常规操作

```bash
cd /home/young/ros2_ws
source install/setup.bash
```

每次构建用 `--symlink-install`，改 Python 不需要重建：

```bash
colcon build --packages-select franka_telep --symlink-install
colcon build --packages-select franka_policy_controller franka_policy_runtime --symlink-install
```

测试：

```bash
colcon test --packages-select franka_telep
colcon test-result --verbose

# Python-only 快速测试
PYTHONPATH=src/franka_telep pytest src/franka_telep/test/ -q
```

## 控制路径

| 输入 | 控制器模式 | 输出 topic |
|---|---|---|
| 6D TCP delta + 夹爪 | `cartesian_delta` / `policy_twist_controller` | `/policy/cartesian_delta` (Twist) |
| 7 个关节角 | `joint_position` / `policy_joint_controller` | `/policy/joint_target` (JointState) |
| uArm 物理遥操作 | 跟随控制器 (impedance) | `/uarm_leader/joint_states` |
| 真机 base | `robot_base.launch.py` | JTC + 关节 broadcast |

真机启动：

```bash
ros2 launch franka_policy_runtime robot_base.launch.py \
  robot_ip:=172.16.0.2 \
  load_gripper:=true \
  controller_mode:=cartesian_delta
```

## uArm 遥操作 + 数据采集

完整遥操作启动：

```bash
ros2 launch franka_telep uarm_openvla_collect.launch.py \
  robot_ip:=172.16.0.2 \
  instruction:="pick up the red block"
```

采集控制：

```bash
# 保存当前 episode
ros2 topic pub --once /franka_teleop/dataset_recording \
  std_msgs/msg/Bool "{data: false}"

# 设置下一轮指令
ros2 topic pub --once /franka_teleop/dataset_instruction \
  std_msgs/msg/String "{data: 'place the block'}"
```

数据默认保存位置：

```text
~/franka_openvla_data/franka_teleop/raw/episode_XXXXXX/
```

转换到 TFDS/RLDS：

```bash
export FRANKA_TELEOP_RAW_DIR=~/franka_openvla_data/franka_teleop/raw
cd install/franka_telep/share/franka_telep/openvla_rlds/franka_teleop_dataset
tfds build --overwrite --data_dir ~/tensorflow_datasets
```

训练数据字段：`observation.image [256,256,3]`、`observation.state [8]`、`action [7]`、`language_instruction`。

## 当前 Home 位置

FR3 第四关节已下调至 `-2.95 rad`：

```text
FR3_READY_JOINTS = [-0.0059, -1.4723, 0.0059, -2.95, -0.0028, 1.6555, 0.8048]
```

Python 默认值在 `franka_mapping.py`，YAML 在 `config/franka_telep.yaml`，三处（franka_teleop、urdf_preview、uarm_leader、franka_home_initializer）一致。

## 控制器架构

- **`TwistIKController`** — C++ ros2_control 插件。两种模式：`cartesian_delta`（Twist→IK→阻抗）和 `joint_position`（直接关节→阻抗）。实时循环内做 KDL IK，PD + Coriolis 补偿输出力矩，带力矩速率饱和。
- **`UarmFollowerController`** — C++ ros2_control 插件。纯 PD 关节阻抗跟随，订阅 `/uarm_leader/joint_states`，无 Coriolis 补偿。
- **`PolicyRuntimeBase`** — Python 模板方法基类，`base_node.py`。子类：`VLAPolicyRuntime`、`BCCubeStackPolicyRuntime`、`AnyGraspRuntime`、`ActionTesterRuntime`。

## 重要已知问题

- `policy_twist_controller` 配置报 `"Node '/policy_twist_controller' has already been added to an executor"` — 由同步参数客户端导致，当前已修复为 `AsyncParametersClient`。
- `controller_mode:=cartesian_delta` 时 YAML 必须配置为 `TwistIKController` 的 `cartesian_delta` 模式。
- RealSense D435I 在 USB 2.1 下默认回退到 `640×480×15`，`--show-args` 可验证参数。
- 夹爪状态不要依赖合并后的 `/joint_states`，应直接订阅 `/fr3_gripper/joint_states`。
