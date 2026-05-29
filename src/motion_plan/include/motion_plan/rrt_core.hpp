#pragma once

#include "motion_plan/rrt_utils.hpp"

#include <moveit/robot_state/robot_state.h>

#include <chrono>
#include <cmath>
#include <functional>
#include <random>
#include <utility>
#include <vector>

namespace motion_plan
{

struct RRTParameters
{
  double range{ 0.25 };
  double min_step_size{ 0.10 };
  double goal_bias{ 0.2 };
  double max_planning_time{ 0.0 };
  int max_iterations{ 2500 };
  double goal_tolerance{ 0.2 };
  double interpolation_step{ 0.1 };
  double collision_check_resolution{ 0.05 };
  bool simplify_path{ false };
  int smoothing_iterations{ 100 };
  bool adaptive_step_size{ true };
  double adaptive_clearance_near{ 0.05 };
  double adaptive_clearance_far{ 0.20 };
};

class RRTCore
{
public:
  struct Node
  {
    moveit::core::RobotState state;
    int parent_index;

    Node(const moveit::core::RobotState& node_state, int parent)
      : state(node_state), parent_index(parent)
    {
    }
  };

  RRTCore(const moveit::core::JointModelGroup* joint_model_group, RRTParameters parameters,
          std::function<double(const moveit::core::RobotState&)> clearance_estimator = {});

  template <typename StateValidator, typename GoalChecker, typename GoalSampler>
  bool solve(const moveit::core::RobotState& start_state, const moveit::core::RobotState& goal_state,
             StateValidator&& is_state_valid, GoalChecker&& is_goal_satisfied, GoalSampler&& sample_goal,
             std::vector<moveit::core::RobotState>& solution) const;

  template <typename StateValidator, typename GoalChecker, typename GoalSampler, typename StopChecker>
  bool solve(const moveit::core::RobotState& start_state, const moveit::core::RobotState& goal_state,
             StateValidator&& is_state_valid, GoalChecker&& is_goal_satisfied, GoalSampler&& sample_goal,
             StopChecker&& should_stop, std::vector<moveit::core::RobotState>& solution) const;

private:
  std::size_t findNearestNodeIndex(const std::vector<Node>& tree, const moveit::core::RobotState& target_state) const;
  moveit::core::RobotState steer(const moveit::core::RobotState& from_state,
                                 const moveit::core::RobotState& to_state) const;
  bool isSegmentValid(const moveit::core::RobotState& from_state, const moveit::core::RobotState& to_state,
                      const std::function<bool(const moveit::core::RobotState&)>& is_state_valid) const;
  void buildSolutionPath(const std::vector<Node>& tree, std::size_t goal_index,
                         std::vector<moveit::core::RobotState>& solution) const;
  std::pair<std::size_t, std::size_t> sampleShortcutPair(std::size_t path_size) const;
  double adaptiveStepSize(const moveit::core::RobotState& from_state) const;

  const moveit::core::JointModelGroup* joint_model_group_;
  RRTParameters parameters_;
  std::function<double(const moveit::core::RobotState&)> clearance_estimator_;
  mutable std::mt19937 rng_;
};

template <typename StateValidator, typename GoalChecker, typename GoalSampler>
bool RRTCore::solve(const moveit::core::RobotState& start_state, const moveit::core::RobotState& goal_state,
                    StateValidator&& is_state_valid, GoalChecker&& is_goal_satisfied, GoalSampler&& sample_goal,
                    std::vector<moveit::core::RobotState>& solution) const
{
  auto no_stop = []() {
    return false;
  };
  return solve(start_state, goal_state, std::forward<StateValidator>(is_state_valid),
               std::forward<GoalChecker>(is_goal_satisfied), std::forward<GoalSampler>(sample_goal), no_stop,
               solution);
}

template <typename StateValidator, typename GoalChecker, typename GoalSampler, typename StopChecker>
bool RRTCore::solve(const moveit::core::RobotState& start_state, const moveit::core::RobotState& goal_state,
                    StateValidator&& is_state_valid, GoalChecker&& is_goal_satisfied, GoalSampler&& sample_goal,
                    StopChecker&& should_stop, std::vector<moveit::core::RobotState>& solution) const
{
  solution.clear();

  if (!is_state_valid(start_state) || !is_state_valid(goal_state))
  {
    return false;
  }

  const auto start_time = std::chrono::steady_clock::now();
  std::vector<Node> tree;
  tree.emplace_back(start_state, -1);

  for (int iteration = 0; iteration < parameters_.max_iterations; ++iteration)
  {
    if (should_stop())
    {
      return false;
    }

    if (parameters_.max_planning_time > 0.0)
    {
      const double elapsed_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
      if (elapsed_seconds >= parameters_.max_planning_time)
      {
        return false;
      }
    }

    moveit::core::RobotState sampled_state = sample_goal(iteration, goal_state);
    const std::size_t nearest_index = findNearestNodeIndex(tree, sampled_state);
    moveit::core::RobotState new_state = steer(tree[nearest_index].state, sampled_state);

    if (!isSegmentValid(tree[nearest_index].state, new_state, is_state_valid))
    {
      continue;
    }

    tree.emplace_back(new_state, static_cast<int>(nearest_index));
    const std::size_t new_index = tree.size() - 1;

    if (is_goal_satisfied(tree[new_index].state) && isSegmentValid(tree[new_index].state, goal_state, is_state_valid))
    {
      tree.emplace_back(goal_state, static_cast<int>(new_index));
      buildSolutionPath(tree, tree.size() - 1, solution);
      if (parameters_.simplify_path)
      {
        solution = detail::shortcutPath(
          solution,
          [this, &is_state_valid](const moveit::core::RobotState& from_state, const moveit::core::RobotState& to_state) {
            return isSegmentValid(from_state, to_state, is_state_valid);
          },
          parameters_.smoothing_iterations,
          [this](std::size_t path_size) { return sampleShortcutPair(path_size); });
      }
      return true;
    }
  }

  return false;
}

}  // namespace motion_plan
