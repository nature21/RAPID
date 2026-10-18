"""Task scene for the packaged real2sim topple_scene bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco

from scene_io.runtime_assets import fr3_task_v1_xml


_PACKAGE_NAME = 'topple_scene'
_PACKAGE_DIR = Path(__file__).resolve().parent
SCENE_DIR = _PACKAGE_DIR / "bundle"
SCENE_XML = SCENE_DIR / "scene.xml"
FR3_XML = fr3_task_v1_xml()
CONFIG: dict[str, Any] = {
    "scene_dir": SCENE_DIR,
    "scene_xml": SCENE_XML,
    "fr3_xml": FR3_XML,
    "task_name": f"real2sim_{_PACKAGE_NAME}",
}

DEFAULT_CAMERA: dict[str, Any] = {'camera_lookat': (0.6665984673560982, 0.017515233113547968, -0.006979221333612284), 'camera_distance': 1.1367357223371042, 'camera_azimuth': 90.39710139846308, 'camera_elevation': -37.18300024138371}
FR3_HOME_JOINT_QPOS: dict[str, float] = {'fr3_joint1': -0.0030637685314503847, 'fr3_joint2': -0.7778934457717781, 'fr3_joint3': 0.01585533499243391, 'fr3_joint4': -2.3693353790401894, 'fr3_joint5': -0.004060005459262234, 'fr3_joint6': 1.554097367289531, 'fr3_joint7': 0.7704814906387026, 'finger_joint1': 0.04, 'finger_joint2': 0.04}
FR3_BASE_POS = (0.0, 0.0, 0.0)
FR3_BASE_QUAT = (1.0, 0.0, 0.0, 0.0)
FR3_MOUNT_FRAME = "fr3_mount"


def make_spec() -> mujoco.MjSpec:
    """Compose the reconstruction and standalone FR3 submodel."""

    scene = mujoco.MjSpec.from_file(str(SCENE_XML))
    robot = mujoco.MjSpec.from_file(str(FR3_XML))
    home = robot.add_key(name="home")
    home.qpos = list(FR3_HOME_JOINT_QPOS.values())
    home.ctrl = [
        FR3_HOME_JOINT_QPOS[f"fr3_joint{index}"] for index in range(1, 8)
    ] + [FR3_HOME_JOINT_QPOS["finger_joint1"]]
    mount = scene.worldbody.add_frame(
        name=FR3_MOUNT_FRAME,
        pos=FR3_BASE_POS,
        quat=FR3_BASE_QUAT,
    )
    # Passing the empty prefix preserves the canonical FR3 names. Omitting the
    # argument makes MjSpec add a leading slash to every attached name.
    scene.attach(robot, prefix="", frame=mount)
    return scene


def make_model() -> mujoco.MjModel:
    """Return the compiled reconstruction-plus-FR3 model."""

    return make_spec().compile()


def make_data(model: mujoco.MjModel | None = None) -> mujoco.MjData:
    """Return data with the FR3 at home and its gripper open."""

    model = make_model() if model is None else model
    data = mujoco.MjData(model)
    home_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if home_id < 0:
        raise KeyError("missing FR3 home keyframe")
    mujoco.mj_resetDataKeyframe(model, data, home_id)
    mujoco.mj_forward(model, data)
    return data


def launch_viewer() -> None:
    """Open this composed scene in MuJoCo's interactive viewer."""

    import mujoco.viewer

    model = make_model()
    mujoco.viewer.launch(model, make_data(model))


__all__ = [
    "CONFIG",
    "DEFAULT_CAMERA",
    "FR3_BASE_POS",
    "FR3_BASE_QUAT",
    "FR3_HOME_JOINT_QPOS",
    "FR3_XML",
    "SCENE_DIR",
    "launch_viewer",
    "make_data",
    "make_model",
    "make_spec",
]


if __name__ == "__main__":
    launch_viewer()
