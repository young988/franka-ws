#pragma once

#include <array>
#include <string>
#include <vector>

#include "controller_interface/controller_interface.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "sensor_msgs/msg/joint_state.hpp"

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

namespace franka_policy_controller {

class PolicyPositionController : public controller_interface::ControllerInterface {
 public:
  [[nodiscard]] controller_interface::InterfaceConfiguration command_interface_configuration()
      const override;
  [[nodiscard]] controller_interface::InterfaceConfiguration state_interface_configuration()
      const override;
  controller_interface::return_type update(const rclcpp::Time& time,
                                           const rclcpp::Duration& period) override;
  CallbackReturn on_init() override;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& previous_state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& previous_state) override;

 private:
  static constexpr size_t kNumJoints = 7;

  struct Target {
    std::array<double, kNumJoints> positions{};
    uint64_t sequence{0};
    bool valid{false};
  };

  std::vector<std::string> configured_joint_names() const;
  void target_callback(const sensor_msgs::msg::JointState::SharedPtr msg);

  std::string arm_id_{"fr3"};
  std::vector<std::string> joint_names_;
  std::string target_joint_state_topic_{"~/target_joint_states"};
  double max_joint_velocity_rad_per_sec_{0.25};
  double goal_tolerance_rad_{1.0e-4};
  double minimum_motion_duration_sec_{2.0};

  /// Two-stage critically damped filter (4th order overall).
  /// Stage 1 filters the target step.  Stage 2 filters the output of stage 1.
  /// The cascade gives C³ output — position, velocity, acceleration, and jerk
  /// are all zero at t=0⁺ — so libfranka never sees an acceleration step.
  ///
  /// Each stage:  xₖ₊₁ = xₖ + vₖ·dt
  ///              vₖ₊₁ = vₖ + (ω²·(u − xₖ) − 2ω·vₖ)·dt
  /// ω is chosen adaptively per joint so that peak velocity stays under the limit.
  std::array<double, kNumJoints> stage1_positions_{};
  std::array<double, kNumJoints> stage1_velocities_{};
  std::array<double, kNumJoints> stage2_positions_{};
  std::array<double, kNumJoints> stage2_velocities_{};
  std::array<double, kNumJoints> target_positions_{};

  uint64_t target_sequence_{0};
  uint64_t active_target_sequence_{0};
  bool initialized_{false};
  realtime_tools::RealtimeBuffer<Target> target_buffer_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr target_subscription_;
};

}  // namespace franka_policy_controller
