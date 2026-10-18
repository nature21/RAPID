"""Shared playback adapter for test-scene replay/record entry points.

Every ``test_scenes/{primitive}/{example}/replay.py`` provides:

* ``spec_builder() -> mujoco.MjSpec`` — scene geometry for viser.
* ``default_camera`` — camera tuning for video rendering.
* ``default_output`` — default MP4 filename.

:func:`make_playback` returns a :class:`PlaybackBundle` exposing
``browser``/``mujoco``/``record`` callables that all take
``(result, model, data)`` as positional args. The instantiation's main script
is responsible for constructing ``data`` with the correct initial pose
(including any optimiser-chosen values copied back from
``result.initial_pose`` via the instantiation's mapping helpers).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mujoco

from primitive_core.common import CompileResult
from scene_io.replay import replay_browser as _replay_browser
from scene_io.replay import replay_mujoco as _replay_mujoco


SpecBuilder = Callable[[], mujoco.MjSpec]
CameraDict = dict[str, Any]


@dataclass
class PlaybackBundle:
    """Triple of replay/record callables produced by :func:`make_playback`."""

    browser: Callable[..., None]
    mujoco: Callable[..., None]
    record: Callable[..., None]


def make_playback(
    *,
    spec_builder: SpecBuilder,
    default_camera: CameraDict,
    default_output: str,
    playback_speed: float = 1.0,
) -> PlaybackBundle:
    """Build replay/record entry points for one test scene.

    Args:
        spec_builder: Nullary callable returning the scene's
            :class:`mujoco.MjSpec` (used by the viser renderer).
        default_camera: Camera parameters for video rendering
            (``camera_lookat``, ``camera_distance``, etc.).
        default_output: Default MP4 filename when ``record`` is called
            without ``output_path``.
        playback_speed: Default ``playback_speed`` for the replay
            entry points.
    """

    def browser(
        result: CompileResult,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        port: int = 8080,
        playback_speed: float = playback_speed,
        loop: bool = True,
    ) -> None:
        spec = spec_builder()
        _replay_browser(
            result, model, data, spec,
            port=port, playback_speed=playback_speed, loop=loop,
        )

    def native(
        result: CompileResult,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        playback_speed: float = playback_speed,
        loop: bool = True,
    ) -> None:
        _replay_mujoco(
            result, model, data,
            playback_speed=playback_speed, loop=loop,
        )

    def record(
        result: CompileResult,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        output_path: str | None = None,
        **kwargs: Any,
    ) -> None:
        from scene_io.record import record_video
        dt = float(result.trajectory.times[1] - result.trajectory.times[0])
        cam = {**default_camera, **kwargs}
        record_video(
            model, data, result.best_states, dt,
            output_path or default_output, **cam,
        )

    return PlaybackBundle(browser=browser, mujoco=native, record=record)
