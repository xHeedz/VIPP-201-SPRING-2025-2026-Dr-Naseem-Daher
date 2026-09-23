"""Replay -> AgentNode -> SummaryNode. Recreated by Claude (Anthropic), September 2026."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    args = [
        DeclareLaunchArgument("csv_path"),
        DeclareLaunchArgument("model_path"),
        DeclareLaunchArgument("road_type", default_value="highway"),
        DeclareLaunchArgument("friction", default_value="1.0"),
        DeclareLaunchArgument("latency_log", default_value=""),
    ]
    return LaunchDescription(args + [
        Node(package="vipp_aggressiveness", executable="telemetry_publisher_node",
             parameters=[{"csv_path": LaunchConfiguration("csv_path")}]),
        Node(package="vipp_aggressiveness", executable="agent_node",
             parameters=[{"model_path": LaunchConfiguration("model_path"),
                          "road_type": LaunchConfiguration("road_type"),
                          "friction": ParameterValue(LaunchConfiguration("friction"), value_type=float),
                          "latency_log": LaunchConfiguration("latency_log")}]),
        Node(package="vipp_aggressiveness", executable="summary_node"),
    ])
