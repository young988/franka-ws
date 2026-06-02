#ifndef FRANKA_POLICY_CONTROLLER__FRANKA_CARTESIAN_POSE_CONTROLLER_HPP_
#define FRANKA_POLICY_CONTROLLER__FRANKA_CARTESIAN_POSE_CONTROLLER_HPP_

#include <array>
#include <memory>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rclcpp/subscription.hpp"
#include "realtime_tools/realtime_buffer.hpp"

namespace franka_policy_controller
{

struct CartesianPoseReference
{
  std::array<double, 3> position;
  std::array<double, 4> quat_xyzw;
  rclcpp::Time stamp;
};

class FrankaCartesianPoseController : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;
  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;
  controller_interface::CallbackReturn on_configure(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_activate(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::CallbackReturn on_deactivate(const rclcpp_lifecycle::State & previous_state) override;
  controller_interface::return_type update(const rclcpp::Time & time, const rclcpp::Duration & period) override;

private:
  void reference_callback(const geometry_msgs::msg::PoseStamped::SharedPtr msg);
  std::vector<std::string> command_interface_names() const;
  std::vector<std::string> state_interface_names() const;

  std::string arm_id_;
  double reference_timeout_sec_{0.5};
  realtime_tools::RealtimeBuffer<std::shared_ptr<CartesianPoseReference>> reference_buffer_;
  rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr reference_sub_;
};

}  // namespace franka_policy_controller

#endif  // FRANKA_POLICY_CONTROLLER__FRANKA_CARTESIAN_POSE_CONTROLLER_HPP_
