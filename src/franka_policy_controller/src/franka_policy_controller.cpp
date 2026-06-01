#include "franka_policy_controller/franka_policy_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace franka_policy_controller
{
namespace
{
std::vector<std::string> default_joints()
{
  return {
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7"};
}

std::vector<double> vector_param_or_default(
  const rclcpp_lifecycle::LifecycleNode::SharedPtr & node,
  const std::string & name,
  const std::vector<double> & fallback)
{
  auto value = node->get_parameter(name).as_double_array();
  return value.empty() ? fallback : value;
}

bool all_finite(const std::vector<double> & values)
{
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::isfinite(value);
  });
}
}  // namespace

controller_interface::CallbackReturn FrankaPolicyController::on_init()
{
  joint_names_ = auto_declare<std::vector<std::string>>("joints", default_joints());
  auto_declare<std::vector<double>>("p_gains", {600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0});
  auto_declare<std::vector<double>>("d_gains", {30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0});
  auto_declare<std::vector<double>>("effort_limits", {30.0, 30.0, 30.0, 30.0, 15.0, 12.0, 10.0});
  auto_declare<double>("reference_timeout_sec", 0.5);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
FrankaPolicyController::command_interface_configuration() const
{
  return {
    controller_interface::interface_configuration_type::INDIVIDUAL,
    command_interface_names()};
}

controller_interface::InterfaceConfiguration
FrankaPolicyController::state_interface_configuration() const
{
  return {
    controller_interface::interface_configuration_type::INDIVIDUAL,
    state_interface_names()};
}

controller_interface::CallbackReturn FrankaPolicyController::on_configure(
  const rclcpp_lifecycle::State & previous_state)
{
  (void)previous_state;
  const auto node = get_node();
  joint_names_ = node->get_parameter("joints").as_string_array();
  if (joint_names_.empty())
  {
    RCLCPP_ERROR(node->get_logger(), "Parameter 'joints' must not be empty");
    return controller_interface::CallbackReturn::ERROR;
  }

  const auto default_p = std::vector<double>(joint_names_.size(), 100.0);
  const auto default_d = std::vector<double>(joint_names_.size(), 10.0);
  const auto default_effort = std::vector<double>(joint_names_.size(), 20.0);
  p_gains_ = vector_param_or_default(node, "p_gains", default_p);
  d_gains_ = vector_param_or_default(node, "d_gains", default_d);
  effort_limits_ = vector_param_or_default(node, "effort_limits", default_effort);
  reference_timeout_sec_ = node->get_parameter("reference_timeout_sec").as_double();

  if (
    p_gains_.size() != joint_names_.size() || d_gains_.size() != joint_names_.size() ||
    effort_limits_.size() != joint_names_.size())
  {
    RCLCPP_ERROR(node->get_logger(), "Gain and effort limit vectors must match joint count");
    return controller_interface::CallbackReturn::ERROR;
  }

  reference_sub_ = node->create_subscription<trajectory_msgs::msg::JointTrajectory>(
    "~/reference",
    rclcpp::SystemDefaultsQoS(),
    [this](trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
      reference_callback(std::move(msg));
    });
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FrankaPolicyController::on_activate(
  const rclcpp_lifecycle::State & previous_state)
{
  (void)previous_state;
  std::vector<double> positions;
  std::vector<double> velocities;
  if (read_state(positions, velocities))
  {
    auto reference = std::make_shared<JointReference>();
    reference->positions = positions;
    reference->stamp = get_node()->now();
    reference_buffer_.writeFromNonRT(reference);
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FrankaPolicyController::on_deactivate(
  const rclcpp_lifecycle::State & previous_state)
{
  (void)previous_state;
  for (auto & command_interface : command_interfaces_)
  {
    command_interface.set_value(0.0);
  }
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type FrankaPolicyController::update(
  const rclcpp::Time & time, const rclcpp::Duration & period)
{
  (void)period;
  std::vector<double> positions;
  std::vector<double> velocities;
  if (!read_state(positions, velocities))
  {
    return controller_interface::return_type::ERROR;
  }

  auto reference_ptr = reference_buffer_.readFromRT();
  const auto reference = reference_ptr ? *reference_ptr : nullptr;
  const bool reference_valid =
    reference && reference->positions.size() == joint_names_.size() &&
    (time - reference->stamp).seconds() <= reference_timeout_sec_;

  const auto & target = reference_valid ? reference->positions : positions;
  for (std::size_t i = 0; i < joint_names_.size(); ++i)
  {
    const double position_error = target[i] - positions[i];
    const double velocity_error = -velocities[i];
    double effort = p_gains_[i] * position_error + d_gains_[i] * velocity_error;
    const double limit = std::abs(effort_limits_[i]);
    effort = std::clamp(effort, -limit, limit);
    command_interfaces_[i].set_value(effort);
  }
  return controller_interface::return_type::OK;
}

void FrankaPolicyController::reference_callback(
  const trajectory_msgs::msg::JointTrajectory::SharedPtr msg)
{
  if (!msg || msg->points.empty())
  {
    return;
  }
  if (msg->joint_names != joint_names_)
  {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 2000,
      "Rejected reference with unexpected joint order");
    return;
  }

  const auto & positions = msg->points.front().positions;
  if (positions.size() != joint_names_.size() || !all_finite(positions))
  {
    RCLCPP_WARN_THROTTLE(
      get_node()->get_logger(), *get_node()->get_clock(), 2000,
      "Rejected malformed or non-finite reference");
    return;
  }

  auto reference = std::make_shared<JointReference>();
  reference->positions = positions;
  reference->stamp = get_node()->now();
  reference_buffer_.writeFromNonRT(reference);
}

bool FrankaPolicyController::read_state(
  std::vector<double> & positions,
  std::vector<double> & velocities) const
{
  const auto n = joint_names_.size();
  if (state_interfaces_.size() != 2 * n || command_interfaces_.size() != n)
  {
    return false;
  }
  positions.resize(n);
  velocities.resize(n);
  for (std::size_t i = 0; i < n; ++i)
  {
    positions[i] = state_interfaces_[i].get_value();
    velocities[i] = state_interfaces_[i + n].get_value();
    if (!std::isfinite(positions[i]) || !std::isfinite(velocities[i]))
    {
      return false;
    }
  }
  return true;
}

std::vector<std::string> FrankaPolicyController::command_interface_names() const
{
  std::vector<std::string> names;
  names.reserve(joint_names_.size());
  for (const auto & joint : joint_names_)
  {
    names.push_back(joint + "/" + hardware_interface::HW_IF_EFFORT);
  }
  return names;
}

std::vector<std::string> FrankaPolicyController::state_interface_names() const
{
  std::vector<std::string> names;
  names.reserve(2 * joint_names_.size());
  for (const auto & joint : joint_names_)
  {
    names.push_back(joint + "/" + hardware_interface::HW_IF_POSITION);
  }
  for (const auto & joint : joint_names_)
  {
    names.push_back(joint + "/" + hardware_interface::HW_IF_VELOCITY);
  }
  return names;
}

}  // namespace franka_policy_controller

PLUGINLIB_EXPORT_CLASS(
  franka_policy_controller::FrankaPolicyController,
  controller_interface::ControllerInterface)
