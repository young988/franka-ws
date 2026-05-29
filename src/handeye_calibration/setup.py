from setuptools import setup
import os
from glob import glob

package_name = 'handeye_calibration'

# Wrapper scripts installed to lib/<pkg>/ so ros2 launch Node can find them.
# ament_python installs entry_points to bin/; ROS 2 launch looks in lib/<pkg>/.
libexec_dir = os.path.join('lib', package_name)

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        (os.path.join('share', package_name, 'launch'),
         glob('launch/*.launch.py')),
        (os.path.join('share', 'ament_index', 'resource_index', 'packages'),
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (libexec_dir, [
            'scripts/aruco_camera_calibrator',
            'scripts/aruco_handeye_calibrator',
            'scripts/sample_collector',
            'scripts/pixel_to_robot',
            'scripts/target_cloud_filter',
            'scripts/handeye_tf_publisher',
        ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='user@example.com',
    description='Hand-eye calibration pipeline using ArUco markers and RealSense D435i',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'aruco_camera_calibrator = handeye_calibration.aruco_camera_calibrator:main',
            'aruco_handeye_calibrator = handeye_calibration.aruco_handeye_calibrator:main',
            'sample_collector = handeye_calibration.sample_collector:main',
            'pixel_to_robot = handeye_calibration.pixel_to_robot:main',
            'target_cloud_filter = handeye_calibration.target_cloud_filter:main',
            'handeye_tf_publisher = handeye_calibration.handeye_tf_publisher:main',
        ],
    },
)
