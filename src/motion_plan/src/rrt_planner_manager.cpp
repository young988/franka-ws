#include "motion_plan/rrt_planner_manager.hpp"

#include "motion_plan/rrt_planning_context.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <string>
#include <vector>

namespace motion_plan
{
namespace
{

constexpr const char* BASELINE_RRT = "RRTBaseline";
constexpr const char* IMPROVED_RRT = "RRTImproved";

RRTParameters makeBaselineParameters()
{
  RRTParameters parameters;
  parameters.range = 0.18;
  parameters.min_step_size = 0.18;
  parameters.goal_bias = 0.1;
  parameters.max_planning_time = 0.0;
  parameters.max_iterations = 2500;
  parameters.goal_tolerance = 0.2;
  parameters.simplify_path = false;
  parameters.adaptive_step_size = false;
  parameters.smoothing_iterations = 0;
  return parameters;
}

std::string normalizePlannerId(const std::string& planner_id)
{
  if (planner_id.empty())
  {
    return IMPROVED_RRT;
  }

  return planner_id;
}

}  // namespace

bool RRTPlannerManager::initialize(const moveit::core::RobotModelConstPtr& model, const rclcpp::Node::SharedPtr& node,
                                   const std::string& parameter_namespace)
{
  robot_model_ = model;
  node_ = node;
  parameter_namespace_ = parameter_namespace;
  logger_ = node_->get_logger().get_child("motion_plan");
  return true;
}

std::string RRTPlannerManager::getDescription() const
{
  return "motion_plan RRT planner";
}

void RRTPlannerManager::getPlanningAlgorithms(std::vector<std::string>& algs) const
{
  algs = { BASELINE_RRT, IMPROVED_RRT };
}

planning_interface::PlanningContextPtr RRTPlannerManager::getPlanningContext(
    const planning_scene::PlanningSceneConstPtr& planning_scene, const planning_interface::MotionPlanRequest& req,
    moveit_msgs::msg::MoveItErrorCodes& error_code) const
{
  const moveit::core::JointModelGroup* joint_model_group = robot_model_->getJointModelGroup(req.group_name);
  if (joint_model_group == nullptr)
  {
    error_code.val = moveit_msgs::msg::MoveItErrorCodes::INVALID_GROUP_NAME;
    return {};
  }

  auto context = std::make_shared<RRTPlanningContext>(getDescription(), req.group_name, robot_model_, joint_model_group,
                                                      logger_, getParametersForGroup(req.group_name, req.planner_id));
  context->setPlanningScene(planning_scene);
  context->setMotionPlanRequest(req);
  error_code.val = moveit_msgs::msg::MoveItErrorCodes::SUCCESS;
  return context;
}

bool RRTPlannerManager::canServiceRequest(const planning_interface::MotionPlanRequest& req) const
{
  return !req.group_name.empty() && robot_model_->hasJointModelGroup(req.group_name);
}

void RRTPlannerManager::setPlannerConfigurations(const planning_interface::PlannerConfigurationMap& pcs)
{
  config_settings_ = pcs;
}

RRTParameters RRTPlannerManager::getParametersForGroup(const std::string& group_name, const std::string& planner_id) const
{
  const std::string normalized_planner_id = normalizePlannerId(planner_id);
  RRTParameters parameters = normalized_planner_id == BASELINE_RRT ? makeBaselineParameters() : RRTParameters{};

  const auto config_it = [&]() {
    const std::vector<std::string> candidate_names = {
      group_name + "[" + planner_id + "]",
      planner_id,
      group_name + "[" + normalized_planner_id + "]",
      normalized_planner_id,
      group_name,
    };

    for (const auto& candidate_name : candidate_names)
    {
      const auto it = config_settings_.find(candidate_name);
      if (it != config_settings_.end())
      {
        return it;
      }
    }

    return config_settings_.end();
  }();

  if (config_it != config_settings_.end())
  {
    const auto& config = config_it->second.config;
    if (const auto range_it = config.find("range"); range_it != config.end())
    {
      parameters.range = std::stod(range_it->second);
    }
    if (const auto min_step_it = config.find("min_step_size"); min_step_it != config.end())
    {
      parameters.min_step_size = std::stod(min_step_it->second);
    }
    if (const auto bias_it = config.find("goal_bias"); bias_it != config.end())
    {
      parameters.goal_bias = std::stod(bias_it->second);
    }
    if (const auto time_it = config.find("max_planning_time"); time_it != config.end())
    {
      parameters.max_planning_time = std::stod(time_it->second);
    }
    if (const auto iteration_it = config.find("max_iterations"); iteration_it != config.end())
    {
      parameters.max_iterations = std::stoi(iteration_it->second);
    }
    if (const auto tolerance_it = config.find("goal_tolerance"); tolerance_it != config.end())
    {
      parameters.goal_tolerance = std::stod(tolerance_it->second);
    }
    if (const auto step_it = config.find("interpolation_step"); step_it != config.end())
    {
      parameters.interpolation_step = std::stod(step_it->second);
    }
    if (const auto resolution_it = config.find("collision_check_resolution"); resolution_it != config.end())
    {
      parameters.collision_check_resolution = std::stod(resolution_it->second);
    }
    if (const auto simplify_it = config.find("simplify_path"); simplify_it != config.end())
    {
      parameters.simplify_path = simplify_it->second == "true";
    }
    if (const auto smoothing_it = config.find("smoothing_iterations"); smoothing_it != config.end())
    {
      parameters.smoothing_iterations = std::stoi(smoothing_it->second);
    }
    if (const auto adaptive_it = config.find("adaptive_step_size"); adaptive_it != config.end())
    {
      parameters.adaptive_step_size = adaptive_it->second == "true";
    }
    if (const auto clearance_near_it = config.find("adaptive_clearance_near"); clearance_near_it != config.end())
    {
      parameters.adaptive_clearance_near = std::stod(clearance_near_it->second);
    }
    if (const auto clearance_far_it = config.find("adaptive_clearance_far"); clearance_far_it != config.end())
    {
      parameters.adaptive_clearance_far = std::stod(clearance_far_it->second);
    }
  }

  return parameters;
}

}  // namespace motion_plan

PLUGINLIB_EXPORT_CLASS(motion_plan::RRTPlannerManager, planning_interface::PlannerManager)
