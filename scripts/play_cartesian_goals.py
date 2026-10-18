#!/usr/bin/env python3
"""Play an explicit RAPID FR3 Cartesian-goal trajectory on real hardware."""

import argparse
import csv
import json

import numpy as np


SCHEMA = "rapid.fr3_cartesian_goals"
SCHEMA_VERSION = 2
CONTROLLER_NAME = "cartesian_impedance_example_controller"
POSITION_TOLERANCE_M = 1e-3
ORIENTATION_TOLERANCE_RAD = 1e-2
INITIAL_MOVE_DURATION_SEC = 3.0
GRIPPER_COMMAND_TOLERANCE_M = 1e-6
TORQUE_COLLISION_THRESHOLDS = [40.0, 40.0, 36.0, 36.0, 32.0, 28.0, 24.0]
FORCE_COLLISION_THRESHOLDS = [40.0, 40.0, 40.0, 50.0, 50.0, 50.0]


def cmdline_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goals", help="Version-2 RAPID Cartesian-goal JSON")
    parser.add_argument(
        "-o",
        "--output",
        default="cartesian_pose_tracking.csv",
        help="Tracking CSV path (default: %(default)s)",
    )
    return parser.parse_args()


def _finite_number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if not np.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def load_document(path):
    """Load and validate the sole supported deployment schema."""

    with open(path, "r", encoding="utf-8") as goals_file:
        payload = json.load(goals_file)
    if not isinstance(payload, dict):
        raise ValueError("the JSON root must be a version-2 document object")
    if payload.get("schema") != SCHEMA or payload.get("version") != SCHEMA_VERSION:
        raise ValueError(f"expected {SCHEMA!r} version {SCHEMA_VERSION}")
    if payload.get("pose_frame") != "fr3_link0":
        raise ValueError("pose_frame must be 'fr3_link0'")
    if payload.get("controlled_frame") != "fr3_hand_tcp":
        raise ValueError("controlled_frame must be 'fr3_hand_tcp'")
    if payload.get("quaternion_order") != "xyzw":
        raise ValueError("quaternion_order must be 'xyzw'")
    if payload.get("time_semantics") != "zero_order_hold":
        raise ValueError("time_semantics must be 'zero_order_hold'")
    sample_rate = _finite_number(payload.get("sample_rate_hz"), "sample_rate_hz")
    if sample_rate <= 0.0:
        raise ValueError("sample_rate_hz must be positive")

    controller = payload.get("controller")
    if not isinstance(controller, dict):
        raise ValueError("controller parameters are required")
    gripper = payload.get("gripper")
    if not isinstance(gripper, dict):
        raise ValueError("gripper controller parameters are required")
    if gripper.get("width_semantic") != "total":
        raise ValueError("gripper width_semantic must be 'total'")
    try:
        width_range = np.asarray(gripper["range_m"], dtype=float)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("gripper range_m must contain two numbers") from exc
    if (
        width_range.shape != (2,)
        or not np.all(np.isfinite(width_range))
        or width_range[0] < 0.0
        or width_range[0] >= width_range[1]
    ):
        raise ValueError("gripper range_m must be finite and increasing")
    speed = _finite_number(gripper.get("move_speed_m_s"), "move_speed_m_s")
    force = _finite_number(gripper.get("grasp_force_n"), "grasp_force_n")
    maximum_force = _finite_number(
        gripper.get("maximum_force_n"), "maximum_force_n"
    )
    epsilon_inner = _finite_number(
        gripper.get("grasp_epsilon_inner_m"), "grasp_epsilon_inner_m"
    )
    if speed <= 0.0 or force <= 0.0 or maximum_force <= 0.0:
        raise ValueError("gripper speed and forces must be positive")
    if force > maximum_force:
        raise ValueError("grasp_force_n cannot exceed maximum_force_n")
    if epsilon_inner < 0.0:
        raise ValueError("grasp_epsilon_inner_m cannot be negative")
    if gripper.get("grasp_epsilon_outer_policy") != "remaining_physical_opening":
        raise ValueError(
            "grasp_epsilon_outer_policy must be 'remaining_physical_opening'"
        )

    raw_goals = payload.get("goals")
    if not isinstance(raw_goals, list) or not raw_goals:
        raise ValueError("goals must be a non-empty list")
    goals = []
    for index, raw_goal in enumerate(raw_goals):
        if not isinstance(raw_goal, dict):
            raise ValueError(f"goal {index} must be an object")
        try:
            position = np.asarray(raw_goal["position_m"], dtype=float)
            orientation = np.asarray(raw_goal["orientation_xyzw"], dtype=float)
            width = float(raw_goal["gripper_width_m"])
            operation = raw_goal["gripper_operation"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"goal {index} requires position_m, orientation_xyzw, "
                "gripper_width_m, and gripper_operation"
            ) from exc
        if position.shape != (3,) or orientation.shape != (4,):
            raise ValueError(f"goal {index} has invalid pose dimensions")
        if not np.all(np.isfinite(position)) or not np.all(np.isfinite(orientation)):
            raise ValueError(f"goal {index} pose must be finite")
        norm = float(np.linalg.norm(orientation))
        if norm <= 1e-12:
            raise ValueError(f"goal {index} orientation cannot be zero")
        if not np.isfinite(width) or not width_range[0] <= width <= width_range[1]:
            raise ValueError(
                f"goal {index} gripper_width_m must lie within range_m"
            )
        if operation not in {"move", "grasp"}:
            raise ValueError(f"goal {index} has an unknown gripper_operation")
        goals.append(
            {
                "position_m": position,
                "orientation_xyzw": orientation / norm,
                "gripper_width_m": width,
                "gripper_operation": operation,
            }
        )
    for index in range(len(goals) - 1):
        if (
            goals[index]["gripper_operation"] == "grasp"
            and goals[index + 1]["gripper_width_m"]
            > goals[index]["gripper_width_m"] + GRIPPER_COMMAND_TOLERANCE_M
        ):
            raise ValueError(f"GRASP width increases between goals {index} and {index + 1}")
    return {
        "sample_rate_hz": sample_rate,
        "controller": controller,
        "gripper": {
            "minimum_width_m": float(width_range[0]),
            "maximum_width_m": float(width_range[1]),
            "move_speed_m_s": speed,
            "grasp_force_n": force,
            "maximum_force_n": maximum_force,
            "grasp_epsilon_inner_m": epsilon_inner,
        },
        "goals": goals,
    }


def gripper_command_events(goals):
    """Collapse equal-operation monotonic runs without deriving semantics."""

    events = {}
    start = 0
    while start < len(goals):
        operation = goals[start]["gripper_operation"]
        end = start
        direction = 0
        while end + 1 < len(goals):
            following = goals[end + 1]
            if following["gripper_operation"] != operation:
                break
            delta = following["gripper_width_m"] - goals[end]["gripper_width_m"]
            next_direction = (
                0
                if abs(delta) <= GRIPPER_COMMAND_TOLERANCE_M
                else (1 if delta > 0.0 else -1)
            )
            if direction and next_direction and next_direction != direction:
                break
            if next_direction:
                direction = next_direction
            end += 1
        events[start] = (operation, goals[end]["gripper_width_m"])
        start = end + 1
    return events


def issue_gripper_event(gripper, operation, width, parameters):
    """Issue one explicit hardware event using the embedded controller contract."""

    if operation == "grasp":
        gripper.grasp_async(
            width,
            parameters["move_speed_m_s"],
            parameters["grasp_force_n"],
            parameters["grasp_epsilon_inner_m"],
            parameters["maximum_width_m"] - width,
        )
    elif operation == "move":
        gripper.move_async(width, parameters["move_speed_m_s"])
    else:
        raise ValueError("unknown gripper operation")


def pose_matrix(goal, transformations):
    pose = transformations.quaternion_matrix(goal["orientation_xyzw"])
    pose[:3, 3] = goal["position_m"]
    return pose


def pose_error(reference, actual):
    position_error = reference[:3, 3] - actual[:3, 3]
    relative_rotation = actual[:3, :3].T @ reference[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    return position_error, np.arccos(cosine)


def poses_are_close(actual, target):
    position_error, orientation_error = pose_error(target, actual)
    return (
        np.linalg.norm(position_error) <= POSITION_TOLERANCE_M
        and orientation_error <= ORIENTATION_TOLERANCE_RAD
    )


def pose_values(pose, transformations):
    quaternion = transformations.quaternion_from_matrix(pose)
    quaternion /= np.linalg.norm(quaternion)
    return [*pose[:3, 3], *quaternion]


def tracking_header():
    names = ["x_m", "y_m", "z_m", "qx", "qy", "qz", "qw"]
    return [
        "sample",
        "elapsed_sec",
        *(f"reference_{name}" for name in names),
        *(f"actual_{name}" for name in names),
        "error_x_m",
        "error_y_m",
        "error_z_m",
        "position_error_m",
        "orientation_error_rad",
        "gripper_reference_total_width_m",
        "gripper_operation",
    ]


def main():
    args = cmdline_args()
    try:
        document = load_document(args.goals)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise SystemExit(f"Could not load Cartesian goals: {exc}") from exc

    import franka_gripper.msg as gripper_msg
    import rospy
    import tf.transformations as transformations
    from ros_interfaces.control import (
        CartesianInterface,
        ControllerManagerInterface,
    )
    from ros_interfaces.franka import (
        FrankaGripperInterface,
        FrankaHardwareInterface,
        FrankaStateInterface,
    )

    class StreamingGripperInterface(FrankaGripperInterface):
        def __init__(self, namespace="/franka_gripper"):
            super().__init__(namespace)
            if not self._grasp.wait_for_server(timeout=rospy.Duration.from_sec(3)):
                raise RuntimeError("Timed out waiting for gripper grasp server")

        def grasp_async(self, width, speed, force, epsilon_inner, epsilon_outer):
            goal = gripper_msg.GraspGoal()
            goal.width = width
            goal.speed = speed
            goal.force = force
            goal.epsilon = gripper_msg.GraspEpsilon()
            goal.epsilon.inner = epsilon_inner
            goal.epsilon.outer = epsilon_outer
            self._grasp.send_goal(goal)

    goals = document["goals"]
    poses = [pose_matrix(goal, transformations) for goal in goals]
    gripper_parameters = document["gripper"]
    events = gripper_command_events(goals)
    try:
        tracking_file = open(args.output, "w", newline="", encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"Could not open tracking output: {exc}") from exc

    with tracking_file:
        writer = csv.writer(tracking_file)
        writer.writerow(tracking_header())
        rospy.init_node("play_cartesian_goals")
        controller_manager = ControllerManagerInterface([CONTROLLER_NAME])
        was_running = CONTROLLER_NAME in controller_manager.list_running_controllers()
        if was_running:
            controller_manager.stop(CONTROLLER_NAME)
        hardware = FrankaHardwareInterface("/franka_control")
        try:
            response = hardware.set_force_torque_collision_behavior(
                TORQUE_COLLISION_THRESHOLDS, FORCE_COLLISION_THRESHOLDS
            )
            if not response.success:
                raise RuntimeError(f"Failed to set collision thresholds: {response.error}")
        except Exception:
            if was_running:
                controller_manager.start(CONTROLLER_NAME)
            raise
        controller_manager.start(CONTROLLER_NAME)
        cartesian = CartesianInterface(controller_name=CONTROLLER_NAME)
        state = FrankaStateInterface()
        gripper = StreamingGripperInterface()
        rospy.sleep(0.5)

        if not poses_are_close(state.pose, poses[0]):
            cartesian.move_to_pose_w_interpolation(
                state.pose, poses[0], duration_sec=INITIAL_MOVE_DURATION_SEC
            )
        rate = rospy.Rate(document["sample_rate_hz"])
        start_time = rospy.get_time()
        position_errors = []
        orientation_errors = []
        for index, (goal, pose) in enumerate(zip(goals, poses)):
            if rospy.is_shutdown():
                break
            event = events.get(index)
            if event is not None:
                operation, terminal_width = event
                issue_gripper_event(
                    gripper,
                    operation,
                    terminal_width,
                    gripper_parameters,
                )
            cartesian.move_to_pose(pose)
            rate.sleep()
            actual = state.pose
            position_error, orientation_error = pose_error(pose, actual)
            position_error_norm = np.linalg.norm(position_error)
            position_errors.append(position_error_norm)
            orientation_errors.append(orientation_error)
            writer.writerow(
                [
                    index,
                    rospy.get_time() - start_time,
                    *pose_values(pose, transformations),
                    *pose_values(actual, transformations),
                    *position_error,
                    position_error_norm,
                    orientation_error,
                    goal["gripper_width_m"],
                    goal["gripper_operation"],
                ]
            )
            if index % max(1, int(document["sample_rate_hz"] // 2)) == 0:
                tracking_file.flush()

        if position_errors:
            rospy.loginfo(
                "Tracking complete: position RMS %.4f m, max %.4f m; "
                "orientation RMS %.4f rad, max %.4f rad; saved to %s",
                np.sqrt(np.mean(np.asarray(position_errors) ** 2)),
                np.max(position_errors),
                np.sqrt(np.mean(np.asarray(orientation_errors) ** 2)),
                np.max(orientation_errors),
                args.output,
            )
        if not rospy.is_shutdown():
            input("Trajectory complete. Press Enter to open the gripper and go home: ")
            gripper.move(
                gripper_parameters["maximum_width_m"],
                gripper_parameters["move_speed_m_s"],
            )
            if not poses_are_close(state.pose, poses[0]):
                cartesian.move_to_pose_w_interpolation(
                    state.pose,
                    poses[0],
                    duration_sec=INITIAL_MOVE_DURATION_SEC,
                )


if __name__ == "__main__":
    main()
