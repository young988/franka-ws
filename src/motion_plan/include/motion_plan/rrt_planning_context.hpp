#pragma once

#include "motion_plan/rrt_core.hpp"

#include <moveit/planning_interface/planning_interface.h>
#include <moveit/robot_model/robot_model.h>
#include <rclcpp/rclcpp.hpp>

namespace motion_plan
{

class RRTPlanningContext : public planning_interface::PlanningContext
{
public:
  RRTPlanningContext(const std::string& name, const std::string& group,
                     const moveit::core::RobotModelConstPtr& robot_model,
                     const moveit::core::JointModelGroup* joint_model_group, const rclcpp::Logger& logger,
                     RRTParameters parameters);

  bool solve(planning_interface::MotionPlanResponse& res) override;
  bool solve(planning_interface::MotionPlanDetailedResponse& res) override;
  bool terminate() override;
  void clear() override;

private:
  bool solveInternal(planning_interface::MotionPlanResponse& res);
  bool resolveGoalState(const moveit::core::RobotState& start_state, moveit::core::RobotState& goal_state,
                        moveit_msgs::msg::MoveItErrorCodes& error_code) const;
  bool isStateValid(const moveit::core::RobotState& state) const;
  bool isSegmentValid(const moveit::core::RobotState& from_state, const moveit::core::RobotState& to_state) const;
  void loadStartState(moveit::core::RobotState& start_state) const;
  void appendSolution(const std::vector<moveit::core::RobotState>& states,
                      planning_interface::MotionPlanResponse& res) const;

  moveit::core::RobotModelConstPtr robot_model_;
  const moveit::core::JointModelGroup* joint_model_group_;
  rclcpp::Logger logger_;
  RRTParameters parameters_;
  bool terminate_requested_{ false };
};

}  // namespace motion_plan
