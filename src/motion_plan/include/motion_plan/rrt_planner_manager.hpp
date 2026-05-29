#pragma once

#include "motion_plan/rrt_core.hpp"

#include <moveit/planning_interface/planning_interface.h>
#include <moveit/robot_model/robot_model.h>
#include <rclcpp/rclcpp.hpp>

namespace motion_plan
{

class RRTPlannerManager : public planning_interface::PlannerManager
{
public:
  bool initialize(const moveit::core::RobotModelConstPtr& model, const rclcpp::Node::SharedPtr& node,
                  const std::string& parameter_namespace) override;

  std::string getDescription() const override;
  void getPlanningAlgorithms(std::vector<std::string>& algs) const override;
  planning_interface::PlanningContextPtr getPlanningContext(const planning_scene::PlanningSceneConstPtr& planning_scene,
                                                            const planning_interface::MotionPlanRequest& req,
                                                            moveit_msgs::msg::MoveItErrorCodes& error_code) const override;
  bool canServiceRequest(const planning_interface::MotionPlanRequest& req) const override;
  void setPlannerConfigurations(const planning_interface::PlannerConfigurationMap& pcs) override;

private:
  RRTParameters getParametersForGroup(const std::string& group_name, const std::string& planner_id) const;

  moveit::core::RobotModelConstPtr robot_model_;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Logger logger_{ rclcpp::get_logger("motion_plan") };
  std::string parameter_namespace_;
};

}  // namespace motion_plan
