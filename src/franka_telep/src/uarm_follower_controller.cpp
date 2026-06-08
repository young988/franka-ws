#include <cassert>
#include <exception>
#include <string>

#include <franka_telep/uarm_follower_controller.hpp>

namespace
{
constexpr unsigned int NUM_JOINTS = 7;

bool rt_buffer_data_is_valid(const std::shared_ptr<sensor_msgs::msg::JointState> * data)
{
  return data && *data;
}

void set_gravity_compensation(
  std::vector<hardware_interface::LoanedCommandInterface> & command_interfaces)
{
  for (auto & interface : command_interfaces) {
    interface.set_value(0.0);
  }
}
}  // namespace

namespace franka_telep
{

CallbackReturn UarmFollowerController::on_init()
{
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::string>("input_topic", "/uarm_leader/joint_states");
    auto_declare<int64_t>("input_topic_timeout", input_topic_timeout_);
    auto_declare<std::vector<double>>(
      "k_gains", {600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0});
    auto_declare<std::vector<double>>(
      "d_gains", {30.0, 30.0, 30.0, 30.0, 10.0, 10.0, 5.0});
  } catch (const std::exception & e) {
    RCLCPP_ERROR(get_node()->get_logger(), "Exception during init: %s", e.what());
    return CallbackReturn::ERROR;
  }
  leader_joint_state_buffer_ =
    realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>(nullptr);
  return CallbackReturn::SUCCESS;
}

CallbackReturn UarmFollowerController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  input_topic_ = get_node()->get_parameter("input_topic").as_string();
  input_topic_timeout_ = get_node()->get_parameter("input_topic_timeout").as_int();

  const auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  const auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  if (k_gains.size() != NUM_JOINTS || d_gains.size() != NUM_JOINTS) {
    RCLCPP_FATAL(get_node()->get_logger(), "k_gains and d_gains must both have 7 values");
    return CallbackReturn::FAILURE;
  }
  for (unsigned int i = 0; i < NUM_JOINTS; ++i) {
    k_gains_(i) = k_gains[i];
    d_gains_(i) = d_gains[i];
  }
  dq_filtered_.setZero();

  leader_joint_state_subscriber_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
    input_topic_, rclcpp::SystemDefaultsQoS(),
    [this](const sensor_msgs::msg::JointState::SharedPtr msg) {
      if (msg->position.size() == NUM_JOINTS) {
        leader_joint_state_buffer_.writeFromNonRT(msg);
      } else {
        RCLCPP_ERROR(
          get_node()->get_logger(), "Invalid leader JointState size %zu, expected %u",
          msg->position.size(), NUM_JOINTS);
      }
    });

  RCLCPP_INFO(
    get_node()->get_logger(), "UArm follower controller arm_id=%s input_topic=%s",
    arm_id_.c_str(), input_topic_.c_str());

  return CallbackReturn::SUCCESS;
}

CallbackReturn UarmFollowerController::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  update_joint_states();
  initial_q_ = q_;
  dq_filtered_.setZero();
  leader_joint_state_buffer_ =
    realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>(nullptr);
  return CallbackReturn::SUCCESS;
}

CallbackReturn UarmFollowerController::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  leader_joint_state_buffer_ =
    realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>(nullptr);
  set_gravity_compensation(command_interfaces_);
  return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
UarmFollowerController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (unsigned int i = 1; i <= NUM_JOINTS; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/effort");
  }
  return config;
}

controller_interface::InterfaceConfiguration
UarmFollowerController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (unsigned int i = 1; i <= NUM_JOINTS; ++i) {
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/position");
    config.names.push_back(arm_id_ + "_joint" + std::to_string(i) + "/velocity");
  }
  return config;
}

controller_interface::return_type UarmFollowerController::update(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  update_joint_states();
  Vector7d q_goal = initial_q_;

  const auto input = leader_joint_state_buffer_.readFromRT();
  if (rt_buffer_data_is_valid(input)) {
    if (latest_input_is_too_old(input)) {
      set_gravity_compensation(command_interfaces_);
      RCLCPP_ERROR(get_node()->get_logger(), "Latest uArm leader message is too old");
      return controller_interface::return_type::ERROR;
    }

    Eigen::Map<Eigen::VectorXd>(q_goal.data(), NUM_JOINTS) =
      Eigen::Map<Eigen::VectorXd>((*input)->position.data(), NUM_JOINTS);
  }

  constexpr double kAlpha = 0.99;
  dq_filtered_ = (1.0 - kAlpha) * dq_filtered_ + kAlpha * dq_;
  const Vector7d tau = k_gains_.cwiseProduct(q_goal - q_) + d_gains_.cwiseProduct(-dq_filtered_);
  for (unsigned int i = 0; i < NUM_JOINTS; ++i) {
    command_interfaces_[i].set_value(tau(i));
  }

  return controller_interface::return_type::OK;
}

void UarmFollowerController::update_joint_states()
{
  for (unsigned int i = 0; i < NUM_JOINTS; ++i) {
    const auto & position_interface = state_interfaces_.at(2 * i);
    const auto & velocity_interface = state_interfaces_.at(2 * i + 1);

    assert(position_interface.get_interface_name() == "position");
    assert(velocity_interface.get_interface_name() == "velocity");

    q_(i) = position_interface.get_value();
    dq_(i) = velocity_interface.get_value();
  }
}

bool UarmFollowerController::latest_input_is_too_old(
  const std::shared_ptr<sensor_msgs::msg::JointState> * input) const
{
  const rclcpp::Time now = get_node()->get_clock()->now();
  return (now - (*input)->header.stamp).nanoseconds() > input_topic_timeout_;
}

}  // namespace franka_telep

#include "pluginlib/class_list_macros.hpp"
PLUGINLIB_EXPORT_CLASS(franka_telep::UarmFollowerController, controller_interface::ControllerInterface)
