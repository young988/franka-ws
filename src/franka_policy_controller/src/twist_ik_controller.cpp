#include <franka_policy_controller/twist_ik_controller.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <exception>
#include <future>
#include <limits>
#include <utility>
#include <vector>

#include <kdl/frames.hpp>
#include <kdl/tree.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/parameter_client.hpp>

namespace franka_policy_controller
{

CallbackReturn TwistIKController::on_init()
{
  try {
    auto_declare<std::string>("arm_id", "fr3");
    auto_declare<std::string>("command_mode", "cartesian_delta");
    auto_declare<std::string>("command_topic", "/policy/cartesian_delta");
    auto_declare<std::string>("joint_command_topic", "/policy/joint_target");
    auto_declare<std::string>("base_link", "base");
    auto_declare<std::string>("tip_link", "fr3_hand_tcp");
    auto_declare<std::string>("robot_description_node", "robot_state_publisher");
    auto_declare<double>("max_translation_step", max_translation_step_);
    auto_declare<double>("max_rotation_step", max_rotation_step_);
    auto_declare<double>("max_joint_delta", max_joint_delta_);
    auto_declare<double>("max_torque_rate", max_torque_rate_);
    auto_declare<double>("velocity_filter_alpha", velocity_filter_alpha_);
    auto_declare<std::vector<double>>(
      "k_gains", {150.0, 150.0, 150.0, 150.0, 80.0, 50.0, 20.0});
    auto_declare<std::vector<double>>(
      "d_gains", {15.0, 15.0, 15.0, 15.0, 8.0, 5.0, 3.0});
  } catch (const std::exception & exception) {
    RCLCPP_ERROR(get_node()->get_logger(), "Controller initialization failed: %s", exception.what());
    return CallbackReturn::ERROR;
  }
  CartesianDeltaCommand empty_command;
  command_buffer_.writeFromNonRT(empty_command);
  JointPositionCommand empty_joint_command;
  joint_command_buffer_.writeFromNonRT(empty_joint_command);
  return CallbackReturn::SUCCESS;
}

CallbackReturn TwistIKController::on_configure(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  arm_id_ = get_node()->get_parameter("arm_id").as_string();
  command_mode_ = get_node()->get_parameter("command_mode").as_string();
  command_topic_ = get_node()->get_parameter("command_topic").as_string();
  joint_command_topic_ = get_node()->get_parameter("joint_command_topic").as_string();
  base_link_ = get_node()->get_parameter("base_link").as_string();
  tip_link_ = get_node()->get_parameter("tip_link").as_string();
  robot_description_node_ = get_node()->get_parameter("robot_description_node").as_string();
  max_translation_step_ = std::abs(get_node()->get_parameter("max_translation_step").as_double());
  max_rotation_step_ = std::abs(get_node()->get_parameter("max_rotation_step").as_double());
  max_joint_delta_ = std::abs(get_node()->get_parameter("max_joint_delta").as_double());
  max_torque_rate_ = std::abs(get_node()->get_parameter("max_torque_rate").as_double());
  velocity_filter_alpha_ = std::clamp(
    get_node()->get_parameter("velocity_filter_alpha").as_double(), 0.0, 1.0);

  const auto k_gains = get_node()->get_parameter("k_gains").as_double_array();
  const auto d_gains = get_node()->get_parameter("d_gains").as_double_array();
  if (k_gains.size() != kNumJoints || d_gains.size() != kNumJoints) {
    RCLCPP_ERROR(get_node()->get_logger(), "k_gains and d_gains must contain 7 values");
    return CallbackReturn::FAILURE;
  }
  for (size_t index = 0; index < kNumJoints; ++index) {
    k_gains_(index) = k_gains[index];
    d_gains_(index) = d_gains[index];
  }

  franka_robot_model_ = std::make_unique<franka_semantic_components::FrankaRobotModel>(
    arm_id_ + "/" + kRobotModelInterfaceName, arm_id_ + "/" + kRobotStateInterfaceName);

  if (command_mode_ != "cartesian_delta" && command_mode_ != "joint_position") {
    RCLCPP_ERROR(
      get_node()->get_logger(), "command_mode must be cartesian_delta or joint_position, got %s",
      command_mode_.c_str());
    return CallbackReturn::FAILURE;
  }

  if (command_mode_ == "cartesian_delta") {
    if (!load_robot_model() || !build_kinematic_chain()) {
      return CallbackReturn::FAILURE;
    }
    command_subscription_ = get_node()->create_subscription<geometry_msgs::msg::Twist>(
      command_topic_, rclcpp::SystemDefaultsQoS(),
      [this](const geometry_msgs::msg::Twist::SharedPtr message) {
        CartesianDeltaCommand command;
        command.delta = *message;
        command.sequence = next_command_sequence_++;
        command_buffer_.writeFromNonRT(command);
      });
  } else {
    joint_command_subscription_ = get_node()->create_subscription<sensor_msgs::msg::JointState>(
      joint_command_topic_, rclcpp::SystemDefaultsQoS(),
      [this](const sensor_msgs::msg::JointState::SharedPtr message) {
        JointPositionCommand command;
        if (message->name.empty()) {
          if (message->position.size() != kNumJoints) {
            RCLCPP_ERROR(
              get_node()->get_logger(), "Joint target must contain 7 positions, got %zu",
              message->position.size());
            return;
          }
          std::copy(message->position.begin(), message->position.end(), command.positions.begin());
        } else {
          for (size_t index = 0; index < kNumJoints; ++index) {
            const std::string expected_name =
              arm_id_ + "_joint" + std::to_string(index + 1);
            const auto iterator =
              std::find(message->name.begin(), message->name.end(), expected_name);
            if (iterator == message->name.end()) {
              RCLCPP_ERROR(
                get_node()->get_logger(), "Joint target is missing %s", expected_name.c_str());
              return;
            }
            const auto message_index =
              static_cast<size_t>(std::distance(message->name.begin(), iterator));
            if (message_index >= message->position.size()) {
              RCLCPP_ERROR(
                get_node()->get_logger(), "Joint target has no position for %s",
                expected_name.c_str());
              return;
            }
            command.positions[index] = message->position[message_index];
          }
        }
        command.sequence = next_command_sequence_++;
        joint_command_buffer_.writeFromNonRT(command);
      });
  }

  RCLCPP_INFO(
    get_node()->get_logger(),
    "Policy effort controller configured: mode=%s topic=%s",
    command_mode_.c_str(),
    command_mode_ == "cartesian_delta" ? command_topic_.c_str() : joint_command_topic_.c_str());
  return CallbackReturn::SUCCESS;
}

CallbackReturn TwistIKController::on_activate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  franka_robot_model_->assign_loaned_state_interfaces(state_interfaces_);
  update_joint_states();
  q_desired_ = q_;
  dq_filtered_.setZero();
  tau_commanded_.setZero();
  consumed_command_sequence_ = 0;
  CartesianDeltaCommand empty_command;
  command_buffer_.writeFromNonRT(empty_command);
  JointPositionCommand empty_joint_command;
  joint_command_buffer_.writeFromNonRT(empty_joint_command);
  return CallbackReturn::SUCCESS;
}

CallbackReturn TwistIKController::on_deactivate(
  const rclcpp_lifecycle::State & /*previous_state*/)
{
  franka_robot_model_->release_interfaces();
  for (auto & command_interface : command_interfaces_) {
    command_interface.set_value(0.0);
  }
  CartesianDeltaCommand empty_command;
  command_buffer_.writeFromNonRT(empty_command);
  JointPositionCommand empty_joint_command;
  joint_command_buffer_.writeFromNonRT(empty_joint_command);
  return CallbackReturn::SUCCESS;
}

controller_interface::InterfaceConfiguration
TwistIKController::command_interface_configuration() const
{
  controller_interface::InterfaceConfiguration configuration;
  configuration.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (size_t index = 1; index <= kNumJoints; ++index) {
    configuration.names.push_back(
      arm_id_ + "_joint" + std::to_string(index) + "/effort");
  }
  return configuration;
}

controller_interface::InterfaceConfiguration
TwistIKController::state_interface_configuration() const
{
  controller_interface::InterfaceConfiguration configuration;
  configuration.type = controller_interface::interface_configuration_type::INDIVIDUAL;
  for (size_t index = 1; index <= kNumJoints; ++index) {
    configuration.names.push_back(
      arm_id_ + "_joint" + std::to_string(index) + "/position");
    configuration.names.push_back(
      arm_id_ + "_joint" + std::to_string(index) + "/velocity");
  }
  if (franka_robot_model_) {
    const auto model_interfaces = franka_robot_model_->get_state_interface_names();
    configuration.names.insert(
      configuration.names.end(), model_interfaces.begin(), model_interfaces.end());
  }
  return configuration;
}

controller_interface::return_type TwistIKController::update(
  const rclcpp::Time & /*time*/, const rclcpp::Duration & /*period*/)
{
  update_joint_states();

  if (command_mode_ == "cartesian_delta") {
    const auto command = command_buffer_.readFromRT();
    if (command && command->sequence > consumed_command_sequence_) {
      consumed_command_sequence_ = command->sequence;
      if (!apply_cartesian_delta(command->delta)) {
        RCLCPP_WARN(
          get_node()->get_logger(),
          "Rejected Cartesian delta command; holding the previous joint target");
      }
    }
  } else {
    const auto command = joint_command_buffer_.readFromRT();
    if (command && command->sequence > consumed_command_sequence_) {
      consumed_command_sequence_ = command->sequence;
      if (!apply_joint_positions(*command)) {
        RCLCPP_WARN(
          get_node()->get_logger(),
          "Rejected joint position command; holding the previous joint target");
      }
    }
  }

  dq_filtered_ =
    (1.0 - velocity_filter_alpha_) * dq_filtered_ + velocity_filter_alpha_ * dq_;
  const std::array<double, kNumJoints> coriolis_array =
    franka_robot_model_->getCoriolisForceVector();
  const Eigen::Map<const Vector7d> coriolis(coriolis_array.data());
  const Vector7d desired_torque =
    k_gains_.cwiseProduct(q_desired_ - q_) - d_gains_.cwiseProduct(dq_filtered_) + coriolis;
  tau_commanded_ = saturate_torque_rate(desired_torque);

  for (size_t index = 0; index < kNumJoints; ++index) {
    command_interfaces_[index].set_value(tau_commanded_(index));
  }
  return controller_interface::return_type::OK;
}

bool TwistIKController::apply_joint_positions(const JointPositionCommand & command)
{
  for (size_t index = 0; index < kNumJoints; ++index) {
    const double target = command.positions[index];
    if (!std::isfinite(target) || std::abs(target - q_(index)) > max_joint_delta_) {
      RCLCPP_WARN(
        get_node()->get_logger(),
        "Invalid joint target at index %zu: target=%.6f current=%.6f limit=%.6f",
        index, target, q_(index), max_joint_delta_);
      return false;
    }
  }
  for (size_t index = 0; index < kNumJoints; ++index) {
    q_desired_(index) = command.positions[index];
  }
  return true;
}

bool TwistIKController::load_robot_model()
{
  using namespace std::chrono_literals;
  auto parameter_client =
    std::make_shared<rclcpp::AsyncParametersClient>(get_node(), robot_description_node_);
  if (!parameter_client->wait_for_service(5s)) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Parameter service for %s is unavailable",
      robot_description_node_.c_str());
    return false;
  }
  const auto future = parameter_client->get_parameters({"robot_description"});
  if (future.wait_for(5s) != std::future_status::ready) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Timed out reading robot_description from %s",
      robot_description_node_.c_str());
    return false;
  }
  const auto parameters = future.get();
  if (parameters.empty() || parameters.front().get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
    RCLCPP_ERROR(get_node()->get_logger(), "robot_description is unavailable");
    return false;
  }
  if (!urdf_model_.initString(parameters.front().as_string())) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to parse robot_description");
    return false;
  }
  return true;
}

bool TwistIKController::build_kinematic_chain()
{
  KDL::Tree tree;
  if (!kdl_parser::treeFromUrdfModel(urdf_model_, tree)) {
    RCLCPP_ERROR(get_node()->get_logger(), "Failed to convert robot_description to a KDL tree");
    return false;
  }
  if (!tree.getChain(base_link_, tip_link_, chain_)) {
    const std::string fallback_tip = arm_id_ + "_link8";
    if (tip_link_ == fallback_tip || !tree.getChain(base_link_, fallback_tip, chain_)) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Failed to build KDL chain %s -> %s",
        base_link_.c_str(), tip_link_.c_str());
      return false;
    }
    RCLCPP_WARN(
      get_node()->get_logger(), "Tip link %s is unavailable; using %s",
      tip_link_.c_str(), fallback_tip.c_str());
    tip_link_ = fallback_tip;
  }
  if (chain_.getNrOfJoints() != kNumJoints) {
    RCLCPP_ERROR(
      get_node()->get_logger(), "Expected 7 chain joints, got %u", chain_.getNrOfJoints());
    return false;
  }

  q_min_ = KDL::JntArray(kNumJoints);
  q_max_ = KDL::JntArray(kNumJoints);
  q_seed_ = KDL::JntArray(kNumJoints);
  q_result_ = KDL::JntArray(kNumJoints);
  size_t joint_index = 0;
  for (const auto & segment : chain_.segments) {
    const auto & joint = segment.getJoint();
    if (joint.getType() == KDL::Joint::None) {
      continue;
    }
    const auto urdf_joint = urdf_model_.getJoint(joint.getName());
    if (!urdf_joint || !urdf_joint->limits || joint_index >= kNumJoints) {
      RCLCPP_ERROR(
        get_node()->get_logger(), "Missing limits for chain joint %s", joint.getName().c_str());
      return false;
    }
    q_min_(joint_index) = urdf_joint->limits->lower;
    q_max_(joint_index) = urdf_joint->limits->upper;
    ++joint_index;
  }

  fk_solver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(chain_);
  ik_velocity_solver_ = std::make_unique<KDL::ChainIkSolverVel_pinv>(chain_);
  ik_position_solver_ = std::make_unique<KDL::ChainIkSolverPos_NR_JL>(
    chain_, q_min_, q_max_, *fk_solver_, *ik_velocity_solver_, 100, 1.0e-5);
  return true;
}

bool TwistIKController::apply_cartesian_delta(const geometry_msgs::msg::Twist & delta)
{
  Eigen::Vector3d translation(delta.linear.x, delta.linear.y, delta.linear.z);
  const double translation_norm = translation.norm();
  if (translation_norm > max_translation_step_ && translation_norm > 0.0) {
    translation *= max_translation_step_ / translation_norm;
  }

  Eigen::Vector3d rotation_vector(delta.angular.x, delta.angular.y, delta.angular.z);
  double rotation_angle = rotation_vector.norm();
  if (rotation_angle > max_rotation_step_ && rotation_angle > 0.0) {
    rotation_vector *= max_rotation_step_ / rotation_angle;
    rotation_angle = max_rotation_step_;
  }

  for (size_t index = 0; index < kNumJoints; ++index) {
    q_seed_(index) = q_(index);
  }
  KDL::Frame current_pose;
  if (fk_solver_->JntToCart(q_seed_, current_pose) < 0) {
    return false;
  }

  Eigen::Matrix3d current_rotation;
  for (size_t row = 0; row < 3; ++row) {
    for (size_t column = 0; column < 3; ++column) {
      current_rotation(row, column) = current_pose.M(row, column);
    }
  }
  const Eigen::Quaterniond current_orientation(current_rotation);
  Eigen::Quaterniond delta_orientation = Eigen::Quaterniond::Identity();
  if (rotation_angle > 1.0e-12) {
    delta_orientation =
      Eigen::AngleAxisd(rotation_angle, rotation_vector / rotation_angle);
  }
  const Eigen::Quaterniond target_orientation =
    (delta_orientation * current_orientation).normalized();
  const KDL::Frame target_pose(
    KDL::Rotation::Quaternion(
      target_orientation.x(), target_orientation.y(), target_orientation.z(),
      target_orientation.w()),
    current_pose.p + KDL::Vector(translation.x(), translation.y(), translation.z()));

  if (ik_position_solver_->CartToJnt(q_seed_, target_pose, q_result_) < 0) {
    return false;
  }
  for (size_t index = 0; index < kNumJoints; ++index) {
    if (!std::isfinite(q_result_(index)) ||
      std::abs(q_result_(index) - q_(index)) > max_joint_delta_)
    {
      return false;
    }
  }
  for (size_t index = 0; index < kNumJoints; ++index) {
    q_desired_(index) = q_result_(index);
  }
  return true;
}

void TwistIKController::update_joint_states()
{
  for (size_t index = 0; index < kNumJoints; ++index) {
    q_(index) = state_interfaces_.at(2 * index).get_value();
    dq_(index) = state_interfaces_.at(2 * index + 1).get_value();
  }
}

TwistIKController::Vector7d TwistIKController::saturate_torque_rate(
  const Vector7d & desired) const
{
  Vector7d saturated;
  for (size_t index = 0; index < kNumJoints; ++index) {
    const double difference = std::clamp(
      desired(index) - tau_commanded_(index), -max_torque_rate_, max_torque_rate_);
    saturated(index) = tau_commanded_(index) + difference;
  }
  return saturated;
}

}  // namespace franka_policy_controller

PLUGINLIB_EXPORT_CLASS(
  franka_policy_controller::TwistIKController,
  controller_interface::ControllerInterface)
