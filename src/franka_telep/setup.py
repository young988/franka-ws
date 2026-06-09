from glob import glob

from setuptools import find_packages, setup

package_name = "franka_telep"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
        ("share/" + package_name + "/rviz", glob("rviz/*.rviz")),
        ("share/" + package_name + "/urdf", glob("urdf/*.urdf")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="young",
    maintainer_email="young@example.com",
    description="ROS 2 teleoperation bridge from UArm-style servo readers to Franka controllers.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "zhonglin_servo_reader = franka_telep.zhonglin_servo_reader_node:main",
            "franka_home_initializer = franka_telep.franka_home_node:main",
            "franka_teleop = franka_telep.franka_teleop_node:main",
            "urdf_joint_state = franka_telep.urdf_joint_state_node:main",
            "uarm_leader_publisher = franka_telep.uarm_leader_publisher:main",
            "openvla_dataset_recorder = franka_telep.openvla_recorder_node:main",
        ],
    },
)
