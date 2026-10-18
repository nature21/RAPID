# Visualization tools

Run these tools from the repository root through the `rapid` environment wrappers.

## Interactive scene viewer

```bash
# OSC torque control, matching the controller used by optimizer rollouts.
./scripts/rapid-view invention_real2sim/flip_scene

# Direct IK dragging for inspecting poses and scene geometry.
./scripts/rapid-view tasks.invention_real2sim.demo_1_scene --controller kinematic

# A task directory or a Python scene file is also accepted by either mode.
./scripts/rapid-view tasks/invention_real2sim/push_scene --paused
./scripts/rapid-view tasks/invention_real2sim/push_scene/scene.py --controller kinematic
```

Scene references also accept a short name such as `push_scene` or a fully
qualified module such as `tasks.invention_real2sim.push_scene.scene`. With no scene
argument, the viewer opens `invention_real2sim/flip_scene`. An interactive desktop display
is required; the wrapper selects GLFW for this viewer.

Both modes support Ctrl-dragging the selected hand, Space to pause or resume,
and Backspace to reset the scene. Their controls differ:

| Mode | Controls |
| --- | --- |
| `osc` (default) | Drag an equilibrium target and let the torque controller move the arm. The numeric keypad adjusts Cartesian position and orientation; keypad Enter cycles compliance, keypad 0 opens/closes the gripper, and G requests a grasp. Terminal telemetry reports tracking error, forces, and torques. |
| `kinematic` | Move the hand with IK dragging, including while paused. Enter opens/closes the gripper. |

OSC gain, force/torque limit, target step, workspace-radius, and telemetry
options are listed by `./scripts/rapid-view --help` and apply to OSC mode.
Each mode prints its keyboard and mouse controls when it starts.

The implementation is `vistools/view_scene.py`; the wrapper configures the
environment. It can also be run as `MUJOCO_GL=glfw ./scripts/rapid-python -m
vistools.view_scene SCENE`.

## Benchmark scene orbit videos

```bash
./scripts/rapid-python -m vistools.render_sim_benchmark_orbits \
  --strategy 1 --case 1,15 --jobs 2
```

This renders existing benchmark scenes and builds a video gallery. With no
selection, it covers tasks 1–8 and cases 1–50. The default output is
`results/sim_benchmark_orbits/`; `--output` selects another directory.
Existing valid videos are reused unless `--overwrite` is supplied.

## Evaluation gallery index

```bash
./scripts/rapid-python -m vistools.build_suite_runs_gallery \
  --experiment-root suite_runs
```

This builds an index of existing evaluation galleries and their reported
success rates. It does not rerun or rescore evaluations. The default experiment
root is `RAPID_EXPERIMENT_ROOT` or `suite_runs`; `--output` overrides the HTML
destination. Use `--help` on either module for all options.
