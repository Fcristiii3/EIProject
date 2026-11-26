"""Package setup for robot_arm_movement."""

from glob import glob
from setuptools import find_packages, setup

package_name = "robot_arm_movement"

setup(name=package_name, version="0.1.0", packages=find_packages(exclude=["test"]), data_files=[("share/ament_index/resource_index/packages", [f"resource/{package_name}"]), (f"share/{package_name}", ["package.xml"]), (f"share/{package_name}/launch", glob("launch/*launch.[pxy][yma]*")), (f"share/{package_name}/config", glob("config/*.yaml"))], install_requires=["setuptools"], zip_safe=True, maintainer="H-Dorina", maintainer_email="dorinahorvath079@gmail.com", description="Dual arm and gripper movement control with 4-DOF OpenMANIPULATOR-X IK and EIProject camera rhythm-game sequencing.", license="Apache-2.0", extras_require={"test": ["pytest"]}, entry_points={"console_scripts": ["arm_controller_node = robot_arm_movement.arm_controller_node:main"]})
