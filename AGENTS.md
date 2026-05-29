# Repository Guidelines

## Project Structure & Module Organization

This is a ROS 2 colcon workspace. Active source packages live under `src/`.

- `src/motion_plan`: C++17 MoveIt planning plugin with headers in `include/motion_plan/`, implementation in `src/`, YAML parameters in `config/`, helper scripts in `scripts/`, and tests in `test/`.
- `src/handeye_calibration`: Python `ament_python` package. Node implementations live in `handeye_calibration/`, launch files in `launch/`, executable wrappers in `scripts/`, and pytest tests in `test/`.
- `src/franka_ros` and `src/realsense-ros`: vendor ROS packages used by the workspace.
- `src/IsaacLab`: external IsaacLab checkout, ignored by colcon via `COLCON_IGNORE`.

Generated `build/`, `install/`, and `log/` directories are colcon outputs.

## Build, Test, and Development Commands

- `colcon list`: show packages that colcon will discover.
- `colcon build`: build all discovered packages.
- `colcon build --packages-select motion_plan`: build only one package.
- `colcon build --packages-up-to handeye_calibration`: build a package and its dependencies.
- `source install/setup.bash`: overlay the workspace after a successful build.
- `colcon test --packages-select motion_plan handeye_calibration`: run package tests.
- `colcon test-result --verbose`: inspect failing test output.

Run commands from `/home/young/ros2_ws`.

## Coding Style & Naming Conventions

Use existing ROS 2 conventions. C++ code in `motion_plan` uses C++17, two-space CMake indentation, `snake_case` filenames, and package-qualified includes such as `motion_plan/rrt_core.hpp`. Keep warnings clean under `-Wall -Wextra -Wpedantic`.

Python modules use `snake_case.py`, four-space indentation, and explicit `main()` entry points. Launch files follow `*.launch.py`.

## Testing Guidelines

Place C++ tests in `test/` and register them with `ament_add_gtest`. Place Python tests in `test/` with names like `test_<feature>.py`. Prefer focused tests for planning math, launch behavior, calibration transforms, and robot-integration regressions.

## Commit & Pull Request Guidelines

The workspace root currently has no readable Git history, so use concise imperative commits, for example `Add RRT planner comparison test` or `Fix hand-eye transform validation`. Keep vendor package changes separate from local package changes.

Pull requests should include affected packages, commands run, test results, and robot, camera, or simulator assumptions. Include logs or screenshots for RViz, MoveIt, RealSense, or calibration workflow changes.

## Agent-Specific Instructions

Do not revert user changes in this workspace. Avoid editing `build/`, `install/`, `log/`, caches, or vendor sources unless the task explicitly requires it. Prefer scoped changes in `src/motion_plan` and `src/handeye_calibration`.
