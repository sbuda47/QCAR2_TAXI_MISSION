from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'qcar2_taxi_2026'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/qcar2_taxi_stack.launch.py']),
        ('share/' + package_name + '/config', ['config/main_controller.yaml']),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='biust',
    maintainer_email='biust@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
    'console_scripts': [
        'lane_follower = qcar2_taxi_2026.nodes.lane_follower_node:main',
        'traffic_detector = qcar2_taxi_2026.nodes.traffic_detector_node:main',
        'taxi_planner = qcar2_taxi_2026.nodes.taxi_planner_node:main',
        'main_controller = qcar2_taxi_2026.nodes.main_controller_node:main',
        "lane_perception = qcar2_taxi_2026.nodes.lane_perception_node:main",
    ],
},
)
