
from __future__ import annotations

import time

import mujoco
import numpy as np

from primitive_core.common import CompileResult


def _step_to_state(model: mujoco.MjModel, data: mujoco.MjData,
                   state: np.ndarray) -> None:
    nq = model.nq
    data.qpos[:] = state[:nq]
    data.qvel[:] = state[nq:]
    mujoco.mj_forward(model, data)


def replay_browser(
    result: CompileResult,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    spec: mujoco.MjSpec | None = None,
    *,
    port: int = 8080,
    playback_speed: float = 1.0,
    loop: bool = True,
) -> None:
    """Replay the compiled trajectory in a browser-based 3D viewer.

    Uses Judo's own :class:`judo.visualizers.model.ViserMjModel` so the
    primitive visualisation matches the rest of the stack.

    Args:
        result: ``CompileResult`` with ``best_states`` and trajectory
            timing.
        model / data: MuJoCo model and data used to render the scene.
        spec: Optional ``MjSpec``. If given, it is used to construct
            ``ViserMjModel``; otherwise the model's underlying XML is
            exported through ``MjSpec.from_file``-equivalent on a
            temporary file. Passing the spec directly is preferred.
        port: viser server port.
        playback_speed: 1.0 = real-time.
        loop: if True, replay continuously.
    """
    import viser
    from judo.visualizers.model import ViserMjModel

    if spec is None:
        # Fall back to extracting XML and re-parsing through MjSpec.
        xml_str = mujoco.mj_saveLastXML("model.xml", model)  # type: ignore[attr-defined]
        spec = mujoco.MjSpec.from_string(xml_str)  # type: ignore[attr-defined]

    server = viser.ViserServer(port=port)
    viser_model = ViserMjModel(server, spec)

    states = result.best_states
    times = result.trajectory.times
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.005

    print(f"\n  Viser replay running at http://localhost:{port}")
    print(f"  {len(states)} frames, dt={dt:.4f}s, speed={playback_speed}x")
    print(f"  Press Ctrl-C to stop.\n")

    try:
        while True:
            for state in states:
                _step_to_state(model, data, state)
                viser_model.set_data(data)
                time.sleep(dt / playback_speed)

            if not loop:
                print("  Replay complete. Keeping server alive (Ctrl-C to exit).")
                while True:
                    time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n  Stopping replay.")


def replay_mujoco(
    result: CompileResult,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    playback_speed: float = 1.0,
    loop: bool = True,
) -> None:
    """Replay the compiled trajectory in MuJoCo's native viewer."""
    import mujoco.viewer

    states = result.best_states
    times = result.trajectory.times
    dt = float(times[1] - times[0]) if len(times) > 1 else 0.005

    print(f"\n  MuJoCo viewer replay")
    print(f"  {len(states)} frames, dt={dt:.4f}s, speed={playback_speed}x")
    print(f"  Close the viewer window to exit.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for state in states:
                if not viewer.is_running():
                    break
                _step_to_state(model, data, state)
                viewer.sync()
                time.sleep(dt / playback_speed)

            if not loop:
                while viewer.is_running():
                    time.sleep(0.1)
                break
