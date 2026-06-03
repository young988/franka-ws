# Action Dimension Test Runtime — Design

2026-06-03

## Purpose

A test node that validates each action dimension (dx, dy, dz, rx, ry, rz, gripper) by sending single-dimension deltas through the full policy pipeline and recording actual TCP outcomes. No policy server needed — actions are manually constructed.

## Architecture

Inherits `PolicyRuntimeBase`, reusing TF lookup, `apply_tcp_delta`, MoveIt IK, trajectory goal lifecycle, gripper, and mutual exclusion via `_goal_active`. Only three extension points are overridden:

```
ActionTesterRuntime(PolicyRuntimeBase)
  ├── _declare_parameters()  → test sequence, step interval, tolerance, csv path
  ├── _create_observer()     → DummyObserver (always ready, empty payload)
  ├── _request_policy()      → pop next action from test sequence, no HTTP
  └── _control_tick()        → wraps base flow, records pre/post TCP for each step
```

`_control_tick` is overridden to intercept the flow at three points:

1. **Before delta** — record `pre_tcp` from TF
2. **After `apply_tcp_delta`** — record `target_tcp`
3. **After trajectory result** — record `post_tcp` from TF, compute error, log step

The overridden tick delegates to the same base helpers: `_update_observer_tcp_pose`, `_handle_gripper`, `_compute_ik`, `_send_trajectory_goal`. The callback chain (`_trajectory_goal_response_cb` → `_trajectory_result_cb`) is left untouched — the result callback is wrapped to record `post_tcp` before re-triggering the next step.

When all steps are exhausted, the node logs a summary (per-step errors, overall max/p95 error) and shuts down.

## Dummy Observer

```python
class _DummyObserver(BaseObserver):
    def observe(self):
        return BackendObservation(ready=True, payload={})
```

## Test Sequence

ROS param `test_sequence`, a list of 7-dim lists `[dx, dy, dz, rx, ry, rz, gripper]`. Default preset:

```yaml
test_sequence:
  - [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # +x
  - [-0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]  # -x
  - [0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0]   # +y
  - [0.0, -0.02, 0.0, 0.0, 0.0, 0.0, 0.0]  # -y
  - [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0]   # +z
  - [0.0, 0.0, -0.02, 0.0, 0.0, 0.0, 0.0]  # -z
  - [0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0]    # +rx (axis-angle)
  - [0.0, 0.0, 0.0, -0.1, 0.0, 0.0, 0.0]   # -rx
  - [0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0]    # +ry
  - [0.0, 0.0, 0.0, 0.0, -0.1, 0.0, 0.0]   # -ry
  - [0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0]    # +rz
  - [0.0, 0.0, 0.0, 0.0, 0.0, -0.1, 0.0]   # -rz
  - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]    # gripper open
  - [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]   # gripper close
```

Additional params:
- `step_interval_sec` (default 2.0) — wait between steps for robot settling
- `action_scale` (default 0.5) — identical to base, applied before delta composition
- `tolerance_pos_m` (default 0.01) — max acceptable position error for `ok` flag
- `csv_output_path` (default `""`) — if set, write results CSV

## Data Recording Per Step

| Field | Source | When |
|-------|--------|------|
| `step` | counter | — |
| `dim_label` | derived from action (e.g. "+dx", "-ry", "gripper_open") | — |
| `action` | from test sequence | — |
| `pre_tcp_position` | TF `_update_observer_tcp_pose()` | before `apply_tcp_delta` |
| `pre_tcp_quat_xyzw` | same | before |
| `target_tcp_position` | `apply_tcp_delta()` output | after delta compute |
| `target_tcp_quat_xyzw` | same | after |
| `post_tcp_position` | TF | after trajectory result cb |
| `post_tcp_quat_xyzw` | same | after |
| `target_error_pos_m` | `norm(target - post)[:3]` | after |
| `ok` | error < tolerance | after |

Console log per step:
```
[action_test] Step +dx: pre=[0.300,0.000,0.500] target=[0.320,0.000,0.500] post=[0.319,0.001,0.499] err=0.002m OK
```

## Files

| File | Purpose |
|------|---------|
| `src/franka_policy_runtime/franka_policy_runtime/action_test_node.py` | `ActionTesterRuntime` + `main()` |
| `src/franka_policy_runtime/config/action_test.yaml` | Default params |
| `src/franka_policy_runtime/launch/action_test.launch.py` | Launch file |
| `src/franka_policy_runtime/test/test_action_test.py` | Unit tests |

Entry point in `setup.py`: `action_test = franka_policy_runtime.action_test_node:main`

## Launch

```bash
ros2 launch franka_policy_runtime action_test.launch.py \
    use_fake_hardware:=true load_gripper:=false
```

## Safety

- Works with `use_fake_hardware:=true` for dry-run without real robot
- Sequential steps with explicit interval between them
- IK failure → log warning, skip step, continue
- TF unavailable → log warning, skip step, continue
- All steps done → log summary table, shutdown node
