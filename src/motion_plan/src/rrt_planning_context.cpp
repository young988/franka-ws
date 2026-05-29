#include "motion_plan/rrt_planning_context.hpp"

#include "motion_plan/rrt_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <moveit/kinematic_constraints/kinematic_constraint.h>
#include <moveit/kinematic_constraints/utils.h>
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_state/conversions.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/time_optimal_trajectory_generation.h>
#include <random>

namespace motion_plan
{

namespace
{
constexpr double IK_TIMEOUT = 0.05;
}

RRTPlanningContext::RRTPlanningContext(const std::string& name, const std::string& group,
                                       const moveit::core::RobotModelConstPtr& robot_model,
                                       const moveit::core::JointModelGroup* joint_model_group,
                                       const rclcpp::Logger& logger, RRTParameters parameters)
  : planning_interface::PlanningContext(name, group)
  , robot_model_(robot_model)
  , joint_model_group_(joint_model_group)
  , logger_(logger)
  , parameters_(parameters)
{
}

bool RRTPlanningContext::solve(planning_interface::MotionPlanResponse& res)
{
  return solveInternal(res);
}

bool RRTPlanningContext::solve(planning_interface::MotionPlanDetailedResponse& res)
{
  planning_interface::MotionPlanResponse basic_response;
  const bool solved = solveInternal(basic_response);

  res.error_code_ = basic_response.error_code_;
  if (!solved)
  {
    return false;
  }

  res.trajectory_.push_back(basic_response.trajectory_);
  res.description_.push_back("plan");
  res.processing_time_.push_back(basic_response.planning_time_);
  return true;
}

bool RRTPlanningContext::terminate()
{
  terminate_requested_ = true;
  return true;
}

void RRTPlanningContext::clear()
{
  terminate_requested_ = false;
}

bool RRTPlanningContext::solveInternal(planning_interface::MotionPlanResponse& res)
{
  clear();
  const auto start_time = std::chrono::steady_clock::now();

  moveit::core::RobotState start_state(robot_model_);
  loadStartState(start_state);
  start_state.enforceBounds(joint_model_group_);
  start_state.update();

  moveit::core::RobotState goal_state(start_state);
  if (!resolveGoalState(start_state, goal_state, res.error_code_))
  {
    return false;
  }

  const auto clearance_estimator = [this](const moveit::core::RobotState& state) {
    return planning_scene_->distanceToCollisionUnpadded(state);
  };

  RRTCore planner(joint_model_group_, parameters_, clearance_estimator);
  std::vector<moveit::core::RobotState> solution;

  const auto is_state_valid = [this](const moveit::core::RobotState& state) {
    return !terminate_requested_ && isStateValid(state);
  };
  const auto is_goal_satisfied = [this, &goal_state](const moveit::core::RobotState& state) {
    return state.distance(goal_state, joint_model_group_) <= parameters_.goal_tolerance;
  };
  const auto sample_goal = [this](int iteration, const moveit::core::RobotState& target_goal) {
    (void)iteration;
    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> unit_distribution(0.0, 1.0);

    if (parameters_.goal_bias > 0.0 && unit_distribution(rng) < parameters_.goal_bias)
    {
      return target_goal;
    }

    moveit::core::RobotState sample(robot_model_);
    sample.setToRandomPositions(joint_model_group_);
    sample.enforceBounds(joint_model_group_);
    sample.update();
    return sample;
  };
  const auto should_stop = [this]() {
    return terminate_requested_;
  };

  if (!planner.solve(start_state, goal_state, is_state_valid, is_goal_satisfied, sample_goal, should_stop, solution))
  {
    res.error_code_.val = terminate_requested_ ? moveit_msgs::msg::MoveItErrorCodes::PREEMPTED :
                                                 moveit_msgs::msg::MoveItErrorCodes::PLANNING_FAILED;
    return false;
  }

  appendSolution(solution, res);
  res.planning_time_ = std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time).count();
  res.error_code_.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  return true;
}

bool RRTPlanningContext::resolveGoalState(const moveit::core::RobotState& start_state, moveit::core::RobotState& goal_state,
                                          moveit_msgs::msg::MoveItErrorCodes& error_code) const
{
  if (request_.goal_constraints.empty())
  {
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
    return false;
  }

  const auto& constraints = request_.goal_constraints.front();
  goal_state = start_state;

  if (!constraints.joint_constraints.empty())
  {
    std::vector<double> target_positions;
    start_state.copyJointGroupPositions(joint_model_group_, target_positions);

    for (const auto& joint_constraint : constraints.joint_constraints)
    {
      const moveit::core::JointModel* joint_model = robot_model_->getJointModel(joint_constraint.joint_name);
      if (joint_model == nullptr)
      {
        continue;
      }

      const auto& group_joint_models = joint_model_group_->getActiveJointModels();
      const auto joint_it = std::find(group_joint_models.begin(), group_joint_models.end(), joint_model);
      if (joint_it == group_joint_models.end())
      {
        continue;
      }

      const std::size_t index = static_cast<std::size_t>(std::distance(group_joint_models.begin(), joint_it));
      target_positions[index] = joint_constraint.position;
    }

    goal_state.setJointGroupPositions(joint_model_group_, target_positions);
    goal_state.enforceBounds(joint_model_group_);
    goal_state.update();
    return isStateValid(goal_state);
  }

  if (!constraints.position_constraints.empty() && !constraints.orientation_constraints.empty())
  {
    const auto& position_constraint = constraints.position_constraints.front();
    const auto& orientation_constraint = constraints.orientation_constraints.front();

    geometry_msgs::msg::Pose target_pose;
    target_pose.position = position_constraint.constraint_region.primitive_poses.front().position;
    target_pose.orientation = orientation_constraint.orientation;

    goal_state = start_state;
    if (!goal_state.setFromIK(joint_model_group_, target_pose, IK_TIMEOUT))
    {
      error_code.val = moveit_msgs::msg::MoveItErrorCodes::NO_IK_SOLUTION;
      return false;
    }

    goal_state.enforceBounds(joint_model_group_);
    goal_state.update();
    return isStateValid(goal_state);
  }

  kinematic_constraints::KinematicConstraintSet goal_constraints(robot_model_);
  goal_constraints.add(constraints, planning_scene_->getTransforms());
  if (goal_constraints.decide(start_state).satisfied)
  {
    goal_state = start_state;
    return true;
  }

  goal_state = start_state;
  for (int attempt = 0; attempt < 20; ++attempt)
  {
    goal_state.setToRandomPositions(joint_model_group_);
    goal_state.enforceBounds(joint_model_group_);
    goal_state.update();

    if (goal_constraints.decide(goal_state).satisfied && isStateValid(goal_state))
    {
      return true;
    }
  }

  error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GOAL_CONSTRAINTS;
  return false;
}

bool RRTPlanningContext::isStateValid(const moveit::core::RobotState& state) const
{
  if (!state.satisfiesBounds(joint_model_group_))
  {
    return false;
  }

  if (!planning_scene_->isStateColliding(state, getGroupName()))
  {
    return true;
  }

  return false;
}

bool RRTPlanningContext::isSegmentValid(const moveit::core::RobotState& from_state,
                                        const moveit::core::RobotState& to_state) const
{
  const double distance = from_state.distance(to_state, joint_model_group_);
  const double resolution = parameters_.collision_check_resolution > 0.0 ? parameters_.collision_check_resolution : 0.05;
  const int steps = std::max(1, static_cast<int>(std::ceil(distance / resolution)));

  for (int step = 1; step <= steps; ++step)
  {
    moveit::core::RobotState waypoint(from_state);
    from_state.interpolate(to_state, static_cast<double>(step) / static_cast<double>(steps), waypoint,
                           joint_model_group_);
    waypoint.enforceBounds(joint_model_group_);
    waypoint.update();

    if (!isStateValid(waypoint))
    {
      return false;
    }
  }

  return true;
}

void RRTPlanningContext::loadStartState(moveit::core::RobotState& start_state) const
{
  moveit::core::robotStateMsgToRobotState(planning_scene_->getTransforms(), request_.start_state, start_state);
}

void RRTPlanningContext::appendSolution(const std::vector<moveit::core::RobotState>& states,
                                        planning_interface::MotionPlanResponse& res) const
{
  auto trajectory = std::make_shared<robot_trajectory::RobotTrajectory>(robot_model_, getGroupName());
  if (states.empty())
  {
    res.trajectory_ = trajectory;
    res.planning_time_ = 0.0;
    return;
  }

  trajectory->addSuffixWayPoint(states.front(), 0.0);

  const double interpolation_step = parameters_.interpolation_step > 0.0 ? parameters_.interpolation_step : 0.1;

  for (std::size_t index = 1; index < states.size(); ++index)
  {
    const auto& from_state = states[index - 1];
    const auto& to_state = states[index];
    const double distance = from_state.distance(to_state, joint_model_group_);
    const int steps = std::max(1, static_cast<int>(std::ceil(distance / interpolation_step)));

    for (int step = 1; step <= steps; ++step)
    {
      moveit::core::RobotState waypoint(from_state);
      from_state.interpolate(to_state, static_cast<double>(step) / static_cast<double>(steps), waypoint,
                             joint_model_group_);
      waypoint.enforceBounds(joint_model_group_);
      waypoint.update();
      trajectory->addSuffixWayPoint(waypoint, 0.0);
    }
  }

  trajectory_processing::TimeOptimalTrajectoryGeneration time_parameterization;
  time_parameterization.computeTimeStamps(*trajectory, request_.max_velocity_scaling_factor,
                                          request_.max_acceleration_scaling_factor);

  res.trajectory_ = trajectory;
  res.planning_time_ = 0.0;
}

}  // namespace motion_plan
