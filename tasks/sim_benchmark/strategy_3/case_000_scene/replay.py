"""Replay / record entry points for strategy 3, benchmark case 000."""

from __future__ import annotations

from scene_io.playback import make_playback
from tasks.sim_benchmark.strategy_3.case_000_scene.scene import DEFAULT_CAMERA, make_spec


playback = make_playback(
    spec_builder=make_spec,
    default_camera=DEFAULT_CAMERA,
    default_output="sim_benchmark_strategy_3_case_000_replay.mp4",
)

replay_browser = playback.browser
replay_mujoco = playback.mujoco
record_video = playback.record
