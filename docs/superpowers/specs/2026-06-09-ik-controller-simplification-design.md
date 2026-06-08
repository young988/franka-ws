# IK Controller Simplification — Design

**Date:** 2026-06-09
**Branch:** `refactor/cleanup-runtimes`

## Motivation

The current policy runtime (`base_node.py`) uses a complex IK pipeline:

```
Policy delta → apply_tcp_delta() → target TCP pose
  → MoveIt GetPositionIK service (blocking)
  → make_joint_trajectory()
  → FollowJointTrajectory action → joint_trajectory_controller (PID) → effort
```

This has multiple pain points:
1. **MoveIt `/compute_ik` service** — blocking sync wait (`time.sleep(0.01)` spin loop), ROS service latency
2. **FollowJointTrajectory action** — goal handle state machine (`_goal_active` / `_active_goal_handle`)
3. **Four-layer control stack** — runtime → MoveIt → trajectory action → PID controller → effort
4. **TF-based TCP lookup** — runtime looks up current TCP pose via TF for `apply_tcp_delta`, less precise than controller's own state interface

The `JointImpedanceIKController` from `franka_spacemouse` demonstrates a simpler approach:
KDL Newton-Raphson IK + direct impedance torque control at 1000Hz inside the ros2_control node.

## Design

### Architecture

```
Policy inference (Python)                C++ controller (ros2_control, 1000Hz)
────────────────────────                 ─────────────────────────────────────
observe → infer → delta                  update():
  |                                        if has_target:
  v                                          snapshot current cartesian pose
publish Twist(/target_cartesian_delta)       target = snapshot + delta
  [linear=dx,dy,dz]                          solve_ik_(target)
  [angular=r,p,y (RPY)]                      τ = Kp·q_err - Kd·dq + coriolis
                                            else:
                                              τ = 0 (wait)
```

Key simplification: the runtime publishes the raw policy delta **without** composing it onto the current TCP pose. The controller does the composition internally using its own 1000Hz state interface — eliminating the TF lookup, the MoveIt service, and the trajectory action layer.

### C++ Controller Changes (`JointImpedanceIKController`)

**Deleted:**
- `spacemouse_callback`, `spacemouse_sub_`, `transform_velocity_to_world_frame_`
- `desired_linear_position_update_`, `desired_angular_position_update_`, `desired_angular_position_update_quaternion_`
- `arm_mounting_orientation_`, hardcoded `max_linear_pos_update`/`max_angular_pos_update`

**Added:**
- Subscription: `rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr delta_sub_`
  - Topic: `/target_cartesian_delta` (global, both sides use the same topic)
  - `linear.{x,y,z}` = translation delta in base frame (already scaled by runtime)
  - `angular.{x,y,z}` = RPY rotation delta in base frame (already scaled by runtime)
- New members: `target_position_` (Eigen::Vector3d), `target_orientation_` (Eigen::Quaterniond), `has_target_` (bool)
- Parameter: `rotation_format` (string, default `"rpy"`)

**Callback (`delta_callback`):**
```
1. std::tie(orientation_, position_) = franka_cartesian_pose_->getCurrentOrientationAndTranslation()
2. delta_pos = Eigen::Vector3d(msg->linear.x, msg->linear.y, msg->linear.z)
3. delta_rpy = Eigen::Vector3d(msg->angular.x, msg->angular.y, msg->angular.z)
4. target_position_ = position_ + delta_pos
5. target_orientation_ = orientation_ * rpy_to_quat(delta_rpy)
6. has_target_ = true
```
Note: `action_scale` is applied by the runtime BEFORE publishing. The controller receives already-scaled deltas.

**`update()` simplified:**
```
if has_target_:
    solve_ik_(target_position_, target_orientation_)
    τ = compute_torque_command_(q_desired, q_current, q_vel)
else:
    τ = 0
write τ to command interfaces
```

**New parameter:**
| Parameter | Type | Default | Description |
|---|---|---|---|
| `rotation_format` | string | `"rpy"` | Rotation delta format (`"rpy"`) |

### Python Runtime Changes (`base_node.py`)

**Deleted (~130 lines):**
- `_compute_ik()` method — MoveIt service call with blocking spin-wait
- `_send_trajectory_goal()` method — trajectory action send + goal management
- `_trajectory_goal_response_cb` / `_trajectory_result_cb`
- `_goal_active` / `_active_goal_handle` / `_ik_callback_group` / `_control_callback_group`
- `_ik_client` / `_trajectory_client`
- `_target_pose_for_ik_link()` — TF-based IK link offset
- Imports: `GetPositionIK`, `MoveItErrorCodes`, `RobotState`, `FollowJointTrajectory`
- Call to `apply_tcp_delta()` in `_control_tick()`

**Added (~15 lines):**
- `_delta_publisher`: `self.create_publisher(Twist, "/target_cartesian_delta", 10)`
- `_publish_delta(action)` helper: splits action, applies `action_scale`, publishes Twist

**`_control_tick()` simplified to:**
```
observe → infer → publish_delta(action) → handle_gripper → schedule next tick
```

**Deleted parameters:**
- `trajectory_action`, `ik_service`, `ik_request_timeout_sec`, `move_group_name`, `ik_link_name`, `avoid_collisions`, `trajectory_duration_sec`, `max_joint_delta_rad`

**New parameter:**
- `delta_topic` (string, default `"/target_cartesian_delta"`)

### Configuration Changes

`franka_bringup_policy_controllers.yaml`:
```yaml
# Before:
fr3_arm_controller:
  type: joint_trajectory_controller/JointTrajectoryController
  # ... PID gains per joint ...

# After:
joint_impedance_ik_controller:
  type: franka_arm_controllers/JointImpedanceIKController

joint_impedance_ik_controller:
  ros__parameters:
    k_gains: [600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0]
    d_gains: [30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0]
    rotation_format: "rpy"
```

### Subclass Impact

`VLAPolicyRuntime` and `BCCubeStackPolicyRuntime` override only `_declare_parameters()` and `_create_observer()`. No changes needed — the delta publishing is in the base class `_control_tick()`.

`AnyGraspRuntime` overrides the entire control loop (phase machine). It does not use MoveIt IK either. No impact.

## Error Handling

- **Controller starts before first delta**: `has_target_ = false`, outputs zero torque. Robot stays in gravity-compensated position.
- **Stop publishing deltas**: impedance control converges to last target, then zero torque (position reached). Robot holds position naturally.
- **Out-of-reach target**: KDL IK returns error < 0 → log error, skip torque command. Robot maintains current position.

## Migration

1. Build and install the modified `franka_arm_controllers` package
2. Update `franka_bringup_policy_controllers.yaml` to use `JointImpedanceIKController`
3. Launch with the new controller config
4. The runtime will publish Twist deltas; the controller will process them

Rollback: revert controller config to use `joint_trajectory_controller/JointTrajectoryController`. The runtime changes are backward-compatible only with the new controller.
