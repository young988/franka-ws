#include "motion_plan/rrt_core.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <random>

namespace motion_plan
{

RRTCore::RRTCore(const moveit::core::JointModelGroup* joint_model_group, RRTParameters parameters,
                 std::function<double(const moveit::core::RobotState&)> clearance_estimator)
  : joint_model_group_(joint_model_group)
  , parameters_(parameters)
  , clearance_estimator_(std::move(clearance_estimator))
  , rng_(std::random_device{}())
{
}

std::size_t RRTCore::findNearestNodeIndex(const std::vector<Node>& tree,
                                         const moveit::core::RobotState& target_state) const
{
  std::size_t nearest_index = 0;
  double nearest_distance = tree.front().state.distance(target_state, joint_model_group_);

  for (std::size_t index = 1; index < tree.size(); ++index)
  {
    const double candidate_distance = tree[index].state.distance(target_state, joint_model_group_);
    if (candidate_distance < nearest_distance)
    {
      nearest_distance = candidate_distance;
      nearest_index = index;
    }
  }

  return nearest_index;
}

moveit::core::RobotState RRTCore::steer(const moveit::core::RobotState& from_state,
                                        const moveit::core::RobotState& to_state) const
{
  moveit::core::RobotState steered_state(from_state);
  const double distance = from_state.distance(to_state, joint_model_group_);
  const double max_step_size = parameters_.range > 0.0 ? parameters_.range : distance;
  const double step = std::min(adaptiveStepSize(from_state), std::min(max_step_size, distance));
  const double ratio = distance > 1e-9 ? step / distance : 1.0;

  from_state.interpolate(to_state, ratio, steered_state, joint_model_group_);
  steered_state.enforceBounds(joint_model_group_);
  steered_state.update();
  return steered_state;
}

bool RRTCore::isSegmentValid(const moveit::core::RobotState& from_state, const moveit::core::RobotState& to_state,
                             const std::function<bool(const moveit::core::RobotState&)>& is_state_valid) const
{
  const double distance = from_state.distance(to_state, joint_model_group_);
  const int steps = std::max(1, static_cast<int>(std::ceil(distance / parameters_.collision_check_resolution)));

  for (int step = 1; step <= steps; ++step)
  {
    moveit::core::RobotState waypoint(from_state);
    from_state.interpolate(to_state, static_cast<double>(step) / static_cast<double>(steps), waypoint,
                           joint_model_group_);
    waypoint.enforceBounds(joint_model_group_);
    waypoint.update();

    if (!is_state_valid(waypoint))
    {
      return false;
    }
  }

  return true;
}

void RRTCore::buildSolutionPath(const std::vector<Node>& tree, std::size_t goal_index,
                                std::vector<moveit::core::RobotState>& solution) const
{
  std::vector<moveit::core::RobotState> reversed_path;

  for (int index = static_cast<int>(goal_index); index >= 0; index = tree[static_cast<std::size_t>(index)].parent_index)
  {
    reversed_path.push_back(tree[static_cast<std::size_t>(index)].state);
  }

  solution.assign(reversed_path.rbegin(), reversed_path.rend());
}

std::pair<std::size_t, std::size_t> RRTCore::sampleShortcutPair(std::size_t path_size) const
{
  return detail::sampleShortcutPair(path_size, rng_);
}

double RRTCore::adaptiveStepSize(const moveit::core::RobotState& from_state) const
{
  const double max_step_size = parameters_.range > 0.0 ? parameters_.range : parameters_.min_step_size;

  if (!parameters_.adaptive_step_size || !clearance_estimator_)
  {
    return max_step_size;
  }

  const double clearance = clearance_estimator_(from_state);
  return detail::adaptiveStepSizeFromClearance(clearance, parameters_.min_step_size, max_step_size,
                                               parameters_.adaptive_clearance_near,
                                               parameters_.adaptive_clearance_far);
}

}  // namespace motion_plan
