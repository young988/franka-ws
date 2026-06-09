#pragma once

#include <cstdint>
#include <array>
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <controller_interface/controller_interface.hpp>
#include <franka_semantic_components/franka_robot_model.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <kdl/chain.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainiksolverpos_nr_jl.hpp>
#include <kdl/chainiksolvervel_pinv.hpp>
#include <kdl/jntarray.hpp>
#include <rclcpp/rclcpp.hpp>
#include <realtime_tools/realtime_buffer.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <urdf/model.h>

namespace franka_policy_controller
{

using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;

struct CartesianDeltaCommand
{
  geometry_msgs::msg::Twist delta;
  uint64_t sequence{0};
};

struct JointPositionCommand
{
  std::array<double, 7> positions{};
  uint64_t sequence{0};
};

class TwistIKController : public controller_interface::ControllerInterface
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
  bool load_robot_model();
  bool build_kinematic_chain();
  bool apply_cartesian_delta(const geometry_msgs::msg::Twist & delta);
  bool apply_joint_positions(const JointPositionCommand & command);
  void update_joint_states();
  Vector7d saturate_torque_rate(const Vector7d & desired) const;

  static constexpr size_t kNumJoints = 7;
  static constexpr const char * kRobotModelInterfaceName = "robot_model";
  static constexpr const char * kRobotStateInterfaceName = "robot_state";

  std::string arm_id_;
  std::string command_mode_;
  std::string command_topic_;
  std::string joint_command_topic_;
  std::string base_link_;
  std::string tip_link_;
  std::string robot_description_node_;

  double max_translation_step_{0.03};
  double max_rotation_step_{0.20};
  double max_joint_delta_{0.30};
  double max_torque_rate_{1.0};
  double velocity_filter_alpha_{0.99};

  Vector7d q_;
  Vector7d q_desired_;
  Vector7d dq_;
  Vector7d dq_filtered_;
  Vector7d tau_commanded_;
  Vector7d k_gains_;
  Vector7d d_gains_;

  urdf::Model urdf_model_;
  KDL::Chain chain_;
  KDL::JntArray q_min_;
  KDL::JntArray q_max_;
  KDL::JntArray q_seed_;
  KDL::JntArray q_result_;
  std::unique_ptr<KDL::ChainFkSolverPos_recursive> fk_solver_;
  std::unique_ptr<KDL::ChainIkSolverVel_pinv> ik_velocity_solver_;
  std::unique_ptr<KDL::ChainIkSolverPos_NR_JL> ik_position_solver_;

  std::unique_ptr<franka_semantic_components::FrankaRobotModel> franka_robot_model_;
  realtime_tools::RealtimeBuffer<CartesianDeltaCommand> command_buffer_;
  realtime_tools::RealtimeBuffer<JointPositionCommand> joint_command_buffer_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr command_subscription_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_command_subscription_;
  uint64_t next_command_sequence_{1};
  uint64_t consumed_command_sequence_{0};
};

}  // namespace franka_policy_controller
