#ifndef FRANKA_POLICY_CONTROLLER__FRANKA_POLICY_CONTROLLER_HPP_
#define FRANKA_POLICY_CONTROLLER__FRANKA_POLICY_CONTROLLER_HPP_

#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp/subscription.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

namespace franka_policy_controller
{

struct JointReference
{
  std::vector<double> positions;
  rclcpp::Time stamp;
};

class FrankaPolicyController : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;

  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::return_type update(
    const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void reference_callback(const trajectory_msgs::msg::JointTrajectory::SharedPtr msg);

  bool read_state(std::vector<double> & positions, std::vector<double> & velocities) const;

  std::vector<std::string> command_interface_names() const;

  std::vector<std::string> state_interface_names() const;

  std::vector<std::string> joint_names_;
  std::vector<double> p_gains_;
  std::vector<double> d_gains_;
  std::vector<double> effort_limits_;
  double reference_timeout_sec_ = 0.5;

  realtime_tools::RealtimeBuffer<std::shared_ptr<JointReference>> reference_buffer_;
  rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr reference_sub_;
};

}  // namespace franka_policy_controller

#endif  // FRANKA_POLICY_CONTROLLER__FRANKA_POLICY_CONTROLLER_HPP_
