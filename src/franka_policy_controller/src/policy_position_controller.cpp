#include "franka_policy_controller/policy_position_controller.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>

#include "hardware_interface/types/hardware_interface_type_values.hpp"
#include "pluginlib/class_list_macros.hpp"

namespace franka_policy_controller {

controller_interface::InterfaceConfiguration
PolicyPositionController::command_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& joint_name : configured_joint_names()) {
    config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_POSITION);
  }
  return config;
}

controller_interface::InterfaceConfiguration
PolicyPositionController::state_interface_configuration() const {
  controller_interface::InterfaceConfiguration config;
  config.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (const auto& joint_name : configured_joint_names()) {
    config.names.push_back(joint_name + "/" + hardware_interface::HW_IF_POSITION);
  }
  return config;
}

controller_interface::return_type PolicyPositionController::update(
    const rclcpp::Time& /*time*/,
    const rclcpp::Duration& period) {
  const double dt = period.seconds() > 0.0 ? period.seconds() : 0.001;

  if (!initialized_) {
    for (size_t i = 0; i < kNumJoints; ++i) {
      const double pos = state_interfaces_[i].get_value();
      stage1_positions_[i] = pos;
      stage1_velocities_[i] = 0.0;
      stage2_positions_[i] = pos;
      stage2_velocities_[i] = 0.0;
    }
    target_positions_ = stage2_positions_;
    initialized_ = true;
  }

  const Target* target = target_buffer_.readFromRT();

  if (target != nullptr && target->valid &&
      target->sequence != active_target_sequence_) {
    target_positions_ = target->positions;
    active_target_sequence_ = target->sequence;
  }

  // Two-stage critically damped filter (4th order overall).
  //
  // Single stage has acceleration ω²Δ @ t=0⁺ — step discontinuity.
  // Cascading two stages gives C³ output: position, velocity, acceleration,
  // and jerk are all zero at t=0⁺.  libfranka's motion generator only cares
  // about acceleration continuity, so this is safe.
  //
  // Peak velocity of the single-stage step response is ω·Δ / e ≈ 0.368·ω·Δ.
  // Two-stage cascade peak is ≈ 0.264·ω·Δ.  We scale ω accordingly so the
  // net peak stays at max_joint_velocity_rad_per_sec_.
  for (size_t i = 0; i < kNumJoints; ++i) {
    const double error = target_positions_[i] - stage1_positions_[i];
    const double delta = std::max(std::abs(error), 1.0e-10);

    // Single-stage v_peak = ω·Δ/e; two-stage v_peak ≈ 0.264·ω·Δ.
    // Scale so the net v_peak equals max_joint_velocity_rad_per_sec_.
    double omega = max_joint_velocity_rad_per_sec_ / (0.264 * delta);
    omega = std::clamp(omega, 0.3, 300.0);

    // Stage 1: filter the raw target step
    stage1_velocities_[i] +=
        (omega * omega * error - 2.0 * omega * stage1_velocities_[i]) * dt;
    stage1_positions_[i] += stage1_velocities_[i] * dt;

    // Stage 2: filter stage 1 output to get C³ output
    const double error2 = stage1_positions_[i] - stage2_positions_[i];
    stage2_velocities_[i] +=
        (omega * omega * error2 - 2.0 * omega * stage2_velocities_[i]) * dt;
    stage2_positions_[i] += stage2_velocities_[i] * dt;

    command_interfaces_[i].set_value(stage2_positions_[i]);
  }
  return controller_interface::return_type::OK;
}

CallbackReturn PolicyPositionController::on_init() {
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::vector<std::string>>("joints", {});
    auto_declare<std::string>("target_joint_state_topic", "~/target_joint_states");
    auto_declare<double>("max_joint_velocity_rad_per_sec", 0.25);
    auto_declare<double>("goal_tolerance_rad", 1.0e-4);
    auto_declare<double>("minimum_motion_duration_sec", 2.0);
    target_buffer_.writeFromNonRT(Target{});
  } catch (const std::exception& exc) {
    fprintf(stderr, "PolicyPositionController init failed: %s\n", exc.what());
    return CallbackReturn::ERROR;
  }
  return CallbackReturn::SUCCESS;
}

CallbackReturn PolicyPositionController::on_configure(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  joint_names_ = get_node()->get_parameter("joints").as_string_array();
  if (joint_names_.empty()) {
    joint_names_ = configured_joint_names();
  }
  if (joint_names_.size() != kNumJoints) {
    RCLCPP_ERROR(
        get_node()->get_logger(), "Expected %zu joints, got %zu", kNumJoints, joint_names_.size());
    return CallbackReturn::ERROR;
  }

  target_joint_state_topic_ = get_node()->get_parameter("target_joint_state_topic").as_string();
  max_joint_velocity_rad_per_sec_ =
      get_node()->get_parameter("max_joint_velocity_rad_per_sec").as_double();
  goal_tolerance_rad_ = get_node()->get_parameter("goal_tolerance_rad").as_double();
  minimum_motion_duration_sec_ =
      get_node()->get_parameter("minimum_motion_duration_sec").as_double();
  if (max_joint_velocity_rad_per_sec_ <= 0.0 || !std::isfinite(max_joint_velocity_rad_per_sec_)) {
    RCLCPP_ERROR(get_node()->get_logger(), "max_joint_velocity_rad_per_sec must be finite and > 0");
    return CallbackReturn::ERROR;
  }
  if (minimum_motion_duration_sec_ <= 0.0 || !std::isfinite(minimum_motion_duration_sec_)) {
    RCLCPP_ERROR(get_node()->get_logger(), "minimum_motion_duration_sec must be finite and > 0");
    return CallbackReturn::ERROR;
  }

  target_subscription_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      target_joint_state_topic_,
      rclcpp::SystemDefaultsQoS(),
      [this](const sensor_msgs::msg::JointState::SharedPtr msg) { target_callback(msg); });

  RCLCPP_INFO(
      get_node()->get_logger(), "PolicyPositionController listening on %s",
      target_joint_state_topic_.c_str());
  return CallbackReturn::SUCCESS;
}

CallbackReturn PolicyPositionController::on_activate(
    const rclcpp_lifecycle::State& /*previous_state*/) {
  initialized_ = false;
  active_target_sequence_ = 0;
  target_buffer_.writeFromNonRT(Target{});
  return CallbackReturn::SUCCESS;
}

std::vector<std::string> PolicyPositionController::configured_joint_names() const {
  if (!joint_names_.empty()) {
    return joint_names_;
  }
  std::vector<std::string> names;
  names.reserve(kNumJoints);
  for (size_t i = 1; i <= kNumJoints; ++i) {
    names.push_back(arm_id_ + "_joint" + std::to_string(i));
  }
  return names;
}

void PolicyPositionController::target_callback(
    const sensor_msgs::msg::JointState::SharedPtr msg) {
  if (msg->position.size() < kNumJoints) {
    RCLCPP_WARN(
        get_node()->get_logger(), "Ignoring target JointState with %zu positions",
        msg->position.size());
    return;
  }

  Target target;
  if (msg->name.empty()) {
    std::copy_n(msg->position.begin(), kNumJoints, target.positions.begin());
  } else {
    std::unordered_map<std::string, double> by_name;
    const size_t count = std::min(msg->name.size(), msg->position.size());
    for (size_t i = 0; i < count; ++i) {
      by_name[msg->name[i]] = msg->position[i];
    }
    for (size_t i = 0; i < kNumJoints; ++i) {
      const auto found = by_name.find(joint_names_.at(i));
      if (found == by_name.end()) {
        RCLCPP_WARN(
            get_node()->get_logger(), "Ignoring target JointState missing joint %s",
            joint_names_.at(i).c_str());
        return;
      }
      target.positions.at(i) = found->second;
    }
  }

  for (double position : target.positions) {
    if (!std::isfinite(position)) {
      RCLCPP_WARN(get_node()->get_logger(), "Ignoring target JointState with non-finite position");
      return;
    }
  }

  target.valid = true;
  target.sequence = ++target_sequence_;
  target_buffer_.writeFromNonRT(target);
}

}  // namespace franka_policy_controller

PLUGINLIB_EXPORT_CLASS(franka_policy_controller::PolicyPositionController,
                       controller_interface::ControllerInterface)
