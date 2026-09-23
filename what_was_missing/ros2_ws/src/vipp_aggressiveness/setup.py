from glob import glob

from setuptools import setup

package_name = "vipp_aggressiveness"

setup(
    name=package_name,
    version="0.1.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml", "README.md"]),
        ("share/" + package_name + "/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Hadi Al Shmaissani",
    maintainer_email="hadi.shmaissani123@gmail.com",
    description="Aggressiveness index ROS2 nodes, recreated by Claude (Anthropic), September 2026",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "agent_node = vipp_aggressiveness.agent_node:main",
            "telemetry_publisher_node = vipp_aggressiveness.telemetry_publisher_node:main",
            "summary_node = vipp_aggressiveness.summary_node:main",
        ],
    },
)
