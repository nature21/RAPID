"""Replay / record entry points for the real2sim demo_4_scene task."""

from __future__ import annotations

from scene_io.playback import make_playback
from tasks.invention_real2sim.demo_4_scene.scene import DEFAULT_CAMERA, make_spec


playback = make_playback(
    spec_builder=make_spec,
    default_camera=DEFAULT_CAMERA,
    default_output="real2sim_demo_4_scene_replay.mp4",
)

replay_browser = playback.browser
replay_mujoco = playback.mujoco
record_video = playback.record
