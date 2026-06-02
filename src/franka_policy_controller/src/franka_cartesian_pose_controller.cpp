#include "franka_policy_controller/franka_cartesian_pose_controller.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <string>
#include <vector>

#include "geometry_msgs/msg/pose_stamped.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace franka_policy_controller
{

namespace
{
bool all_finite(const std::array<double, 3> & position, const std::array<double, 4> & quat)
{
  return std::all_of(position.begin(), position.end(), [](double v) { return std::isfinite(v); }) &&
         std::all_of(quat.begin(), quat.end(), [](double v) { return std::isfinite(v); });
}

std::array<double, 16> to_column_major_pose(
  const std::array<double, 3> & position,
  const std::array<double, 4> & quat_xyzw)
{
  const double x = quat_xyzw[0];
  const double y = quat_xyzw[1];
  const double z = quat_xyzw[2];
  const double w = quat_xyzw[3];

  return {
    1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + z * w), 2.0 * (x * z - y * w), 0.0,
    2.0 * (x * y - z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z + x * w), 0.0,
    2.0 * (x * z + y * w), 2.0 * (y * z - x * w), 1.0 - 2.0 * (x * x + y * y), 0.0,
    position[0], position[1], position[2], 1.0};
}
}  // namespace

controller_interface::CallbackReturn FrankaCartesianPoseController::on_init()
{
  arm_id_ = auto_declare<std::string>("arm_id", "fr3");
  auto_declare<double>("reference_timeout_sec", 0.5);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration FrankaCartesianPoseController::command_interface_configuration() const
{
  return {controller_interface::interface_configuration_type::INDIVIDUAL, command_interface_names()};
}

controller_interface::InterfaceConfiguration FrankaCartesianPoseController::state_interface_configuration() const
{
  return {controller_interface::interface_configuration_type::INDIVIDUAL, state_interface_names()};
}

controller_interface::CallbackReturn FrankaCartesianPoseController::on_configure(
  const rclcpp_lifecycle::State & previous_state)
{
  (void)previous_state;
  const auto node = get_node();
  arm_id_ = node->get_parameter("arm_id").as_string();
  reference_timeout_sec_ = node->get_parameter("reference_timeout_sec").as_double();
  reference_sub_ = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "~/reference", rclcpp::SystemDefaultsQoS(),
    [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) { reference_callback(std::move(msg)); });
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FrankaCartesianPoseController::on_activate(
  const rclcpp_lifecycle::State & previous_state)
{
  (void)previous_state;
  auto reference = std::make_shared<CartesianPoseReference>();
  reference->position = {state_interfaces_[12].get_value(), state_interfaces_[13].get_value(), state_interfaces_[14].get_value()};
  reference->quat_xyzw = {0.0, 0.0, 0.0, 1.0};
  reference->stamp = get_node()->now();
  reference_buffer_.writeFromNonRT(reference);
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::CallbackReturn FrankaCartesianPoseController::on_deactivate(
  const rclcpp_lifecycle::State & previous_state)
{
  (void)previous_state;
  return controller_interface::CallbackReturn::SUCCESS;
}

controller_interface::return_type FrankaCartesianPoseController::update(
  const rclcpp::Time & time,
  const rclcpp::Duration & period)
{
  (void)period;
  auto reference_ptr = reference_buffer_.readFromRT();
  const auto reference = reference_ptr ? *reference_ptr : nullptr;
  if (!reference || (time - reference->stamp).seconds() > reference_timeout_sec_) {
    return controller_interface::return_type::OK;
  }

  const auto pose = to_column_major_pose(reference->position, reference->quat_xyzw);
  for (std::size_t i = 0; i < pose.size(); ++i) {
    command_interfaces_[i].set_value(pose[i]);
  }
  return controller_interface::return_type::OK;
}

void FrankaCartesianPoseController::reference_callback(
  const geometry_msgs::msg::PoseStamped::SharedPtr msg)
{
  if (!msg) {
    return;
  }
  auto reference = std::make_shared<CartesianPoseReference>();
  reference->position = {msg->pose.position.x, msg->pose.position.y, msg->pose.position.z};
  reference->quat_xyzw = {
    msg->pose.orientation.x,
    msg->pose.orientation.y,
    msg->pose.orientation.z,
    msg->pose.orientation.w,
  };
  if (!all_finite(reference->position, reference->quat_xyzw)) {
    return;
  }
  reference->stamp = get_node()->now();
  reference_buffer_.writeFromNonRT(reference);
}

std::vector<std::string> FrankaCartesianPoseController::command_interface_names() const
{
  std::vector<std::string> names;
  names.reserve(16);
  for (std::size_t i = 0; i < 16; ++i) {
    names.push_back(std::to_string(i) + "/cartesian_pose_command");
  }
  return names;
}

std::vector<std::string> FrankaCartesianPoseController::state_interface_names() const
{
  std::vector<std::string> names;
  names.reserve(16);
  for (std::size_t i = 0; i < 16; ++i) {
    names.push_back(std::to_string(i) + "/cartesian_pose_state");
  }
  return names;
}

}  // namespace franka_policy_controller

PLUGINLIB_EXPORT_CLASS(
  franka_policy_controller::FrankaCartesianPoseController,
  controller_interface::ControllerInterface)
