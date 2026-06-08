#pragma once

#include <memory>
#include <string>

#include <Eigen/Eigen>
#include <controller_interface/controller_interface.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <sensor_msgs/msg/joint_state.hpp>

namespace franka_telep
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

class UarmFollowerController : public controller_interface::ControllerInterface
{
public:
  using Vector7d = Eigen::Matrix<double, 7, 1>;

  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;

  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
  const override;

  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
  const override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void update_joint_states();
  bool latest_input_is_too_old(
    const std::shared_ptr<sensor_msgs::msg::JointState> * input) const;

  std::string arm_id_;
  std::string input_topic_;
  int64_t input_topic_timeout_{200000000};

  Vector7d q_;
  Vector7d initial_q_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d k_gains_;
  Vector7d d_gains_;

  realtime_tools::RealtimeBuffer<std::shared_ptr<sensor_msgs::msg::JointState>>
  leader_joint_state_buffer_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr leader_joint_state_subscriber_;
};

}  // namespace franka_telep
