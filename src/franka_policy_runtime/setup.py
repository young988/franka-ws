from glob import glob

from setuptools import find_packages, setup

package_name = "franka_policy_runtime"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", glob("config/*.yaml")),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="young",
    maintainer_email="young@example.com",
    description="Runtime bridge between policy inference and Franka policy controller",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "vla_policy_runtime = franka_policy_runtime.vla_node:main",
            "bc_cube_stack_runtime = franka_policy_runtime.bc_cube_stack_node:main",
            "action_test_node = franka_policy_runtime.action_test_node:main",
            "action_test = franka_policy_runtime.action_test:main",
        ],
    },
)
