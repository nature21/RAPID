from __future__ import annotations

import unittest
from unittest import mock

import mujoco
import numpy as np

from instantiation_core.runner import build_config, make_parser
from primitive_core.common import (
    AUTO_MAX_PARALLEL_ROLLOUTS,
    GLOBAL_MAX_OPTIMIZATION_BUDGET,
    OptimizationBudget,
    OptimizationPolicy,
    PrimitiveCompilerConfig,
    resolve_parallel_rollouts,
)
from primitive_core.compile import (
    PrimitiveCompileSpec,
    _make_rollout_backend,
    _rollout_in_chunks,
    compile_primitive,
)


_XML = """
<mujoco model="rollout_backend_test">
  <option density="1.2"/>
  <worldbody>
    <body>
      <joint name="slider" type="slide"/>
      <geom type="sphere" size="0.05" friction="0 0 0"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="slider" ctrlrange="-1 1"/>
  </actuator>
</mujoco>
"""


class _FakeWarp:
    @staticmethod
    def is_cuda_available() -> bool:
        return True


class _FakeMJWarpBackend:
    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs


class _RecordingBackend:
    def __init__(self) -> None:
        self.num_threads = 2
        self.x0_batches: list[np.ndarray] = []

    def rollout(
        self,
        x0: np.ndarray,
        controls: np.ndarray,
        last_policy_output: object = None,
    ) -> tuple[np.ndarray, np.ndarray, None]:
        del last_policy_output
        self.x0_batches.append(x0.copy())
        states = np.repeat(x0[:, None, :], controls.shape[1], axis=1)
        sensors = np.empty((x0.shape[0], controls.shape[1], 0))
        return states, sensors, None

    def update(self, num_threads: int) -> None:
        self.num_threads = num_threads


class RolloutBackendTests(unittest.TestCase):
    def test_automatic_parallelism_is_cpu_aware_and_chunk_friendly(self) -> None:
        self.assertEqual(AUTO_MAX_PARALLEL_ROLLOUTS, 48)
        self.assertEqual(
            resolve_parallel_rollouts(128, available_cpus=128),
            32,
        )
        self.assertEqual(
            resolve_parallel_rollouts(192, available_cpus=128),
            48,
        )
        self.assertEqual(
            resolve_parallel_rollouts(128, available_cpus=16),
            16,
        )
        self.assertEqual(
            resolve_parallel_rollouts(128, 40, available_cpus=8),
            40,
        )

    def test_automatic_parallelism_rejects_invalid_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_rollouts"):
            resolve_parallel_rollouts(0, available_cpus=8)
        with self.assertRaisesRegex(ValueError, "max_parallel_rollouts"):
            resolve_parallel_rollouts(8, -1, available_cpus=8)
        with self.assertRaisesRegex(ValueError, "available_cpus"):
            resolve_parallel_rollouts(8, available_cpus=0)

    def test_cli_and_environment_select_rollout_backend(self) -> None:
        parser = make_parser("test")
        args = parser.parse_args(
            ["--rollout-backend", "mjwarp", "--rollout-device", "cuda:1"]
        )

        config = build_config(PrimitiveCompilerConfig, args)

        self.assertEqual(config.rollout_backend, "mjwarp")
        self.assertEqual(config.rollout_device, "cuda:1")

        with mock.patch.dict(
            "os.environ",
            {
                "RAPID_ROLLOUT_BACKEND": "mujoco",
                "RAPID_ROLLOUT_DEVICE": "cuda:2",
            },
        ):
            config = build_config(PrimitiveCompilerConfig, args)

        self.assertEqual(config.rollout_backend, "mujoco")
        self.assertEqual(config.rollout_device, "cuda:2")

    def test_programmatic_config_overrides_cannot_bypass_resource_caps(self) -> None:
        parser = make_parser("test")
        args = parser.parse_args([])
        with mock.patch.dict(
            "os.environ",
            {
                "RAPID_OPTIMIZER_NUM_ROLLOUTS": "16",
                "RAPID_MAX_PARALLEL_ROLLOUTS": "8",
            },
        ):
            config = build_config(
                PrimitiveCompilerConfig,
                args,
                num_rollouts=128,
                max_parallel_rollouts=64,
            )

        self.assertEqual(config.num_rollouts, 16)
        self.assertEqual(config.max_parallel_rollouts, 8)

        with mock.patch.dict("os.environ", {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GLOBAL_MAX"):
                build_config(
                    PrimitiveCompilerConfig,
                    args,
                    num_rollouts=(
                        GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts + 1
                    ),
                )

    def test_reference_budget_is_used_and_cli_accepts_caps(self) -> None:
        parser = make_parser("test")
        args = parser.parse_args(
            [
                "--num-rollouts",
                "32",
                "--max-iters",
                "3",
                "--horizon",
                "1.5",
                "--seed",
                "7",
                "--transit-seed",
                "73",
            ]
        )
        policy = OptimizationPolicy(
            num_nodes=6,
            elite_fraction=0.125,
            horizon=2.5,
            min_iters=4,
            early_stop_patience=2,
        )
        reference = OptimizationBudget(num_rollouts=96, max_iters=8)

        config = build_config(
            PrimitiveCompilerConfig,
            args,
            learned_policy=policy,
            reference_budget=reference,
        )

        self.assertEqual(config.num_rollouts, 32)
        self.assertEqual(config.num_nodes, 6)
        self.assertEqual(config.num_elites, 4)
        self.assertEqual(config.max_iters, 3)
        self.assertEqual(config.horizon, 1.5)
        self.assertEqual(config.min_iters, 3)
        self.assertEqual(config.early_stop_patience, 2)
        self.assertEqual(args.seed, 7)
        self.assertEqual(config.transit_seed, 73)

        smoke_args = parser.parse_args(["--max-iters", "1"])
        smoke = build_config(
            PrimitiveCompilerConfig,
            smoke_args,
            learned_policy=policy,
            reference_budget=reference,
        )
        self.assertEqual(smoke.max_iters, 1)
        self.assertEqual(smoke.min_iters, 1)

    def test_maximum_mode_expands_compute_without_changing_policy(self) -> None:
        parser = make_parser("test")
        args = parser.parse_args([])
        policy = OptimizationPolicy(
            num_nodes=7,
            elite_fraction=0.5,
            horizon=5.5,
            min_iters=4,
            early_stop_patience=2,
        )
        reference = OptimizationBudget(num_rollouts=32, max_iters=6)
        with mock.patch.dict(
            "os.environ",
            {"RAPID_OPTIMIZATION_BUDGET_MODE": "maximum"},
            clear=False,
        ):
            config = build_config(
                PrimitiveCompilerConfig,
                args,
                learned_policy=policy,
                reference_budget=reference,
            )

        self.assertEqual(
            config.num_rollouts,
            GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts,
        )
        self.assertEqual(
            config.max_iters,
            GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters,
        )
        self.assertEqual(config.min_iters, 4)
        self.assertEqual(config.early_stop_patience, 2)
        self.assertEqual(config.num_nodes, 7)
        self.assertEqual(config.num_elites, 48)

    def test_optimization_policy_rejects_noninteger_search_sizes(self) -> None:
        with self.assertRaisesRegex(TypeError, "num_nodes"):
            OptimizationPolicy(  # type: ignore[arg-type]
                num_nodes=4.5,
                elite_fraction=0.25,
                horizon=1.0,
            )

    def test_learned_early_stop_requires_four_minimum_iterations(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_iters >= 4"):
            OptimizationPolicy(
                num_nodes=4,
                elite_fraction=0.25,
                horizon=1.0,
                min_iters=3,
                early_stop_patience=1,
            )

        policy = OptimizationPolicy(
            num_nodes=4,
            elite_fraction=0.25,
            horizon=1.0,
            min_iters=1,
            early_stop_patience=0,
        )
        self.assertEqual(policy.min_iters, 1)

    def test_learned_knot_topology_cannot_be_capped(self) -> None:
        parser = make_parser("test")
        args = parser.parse_args(["--num-nodes", "3"])
        policy = OptimizationPolicy(
            num_nodes=4,
            elite_fraction=0.25,
            horizon=1.0,
        )

        with self.assertRaisesRegex(ValueError, "frozen learned knot topology"):
            build_config(
                PrimitiveCompilerConfig,
                args,
                learned_policy=policy,
                reference_budget=OptimizationBudget(8, 2),
            )

    def test_strategy_phase_budget_can_increase_within_global_limit(self) -> None:
        parser = make_parser("test")
        policy = OptimizationPolicy(
            num_nodes=4,
            elite_fraction=0.25,
            horizon=1.0,
        )
        config = build_config(
            PrimitiveCompilerConfig,
            parser.parse_args([]),
            learned_policy=policy,
            reference_budget=OptimizationBudget(32, 6),
            requested_budget=OptimizationBudget(64, 10),
        )

        self.assertEqual(config.num_rollouts, 64)
        self.assertEqual(config.num_elites, 16)
        self.assertEqual(config.max_iters, 10)

        with self.assertRaisesRegex(ValueError, "global|GLOBAL"):
            build_config(
                PrimitiveCompilerConfig,
                parser.parse_args([]),
                learned_policy=policy,
                reference_budget=OptimizationBudget(32, 6),
                requested_budget=OptimizationBudget(
                    GLOBAL_MAX_OPTIMIZATION_BUDGET.num_rollouts + 1,
                    GLOBAL_MAX_OPTIMIZATION_BUDGET.max_iters,
                ),
            )

    def test_early_stop_requires_predicate_success_and_cost_plateau(self) -> None:
        model = mujoco.MjModel.from_xml_string(_XML)
        data = mujoco.MjData(model)
        nominal = np.zeros((2, 1), dtype=float)
        config = PrimitiveCompilerConfig(
            num_rollouts=1,
            max_parallel_rollouts=1,
            num_nodes=2,
            num_elites=1,
            max_iters=8,
            min_iters=2,
            early_stop_patience=2,
            early_stop_cost_tolerance=1e-8,
            sigma_min=1e-6,
            sigma_max=1e-5,
            horizon=0.02,
            optimize_initial_pose=False,
        )

        result, _ = compile_primitive(
            PrimitiveCompileSpec(
                model=model,
                initial_state=np.concatenate((data.qpos, data.qvel)),
                nominal_knots=nominal,
                sigma_init=np.zeros_like(nominal),
                cost_fn=lambda states, controls: np.zeros(states.shape[0]),
                effect_predicate=lambda states, controls: np.ones(
                    states.shape[0],
                    dtype=bool,
                ),
            ),
            config,
            verbose=False,
        )

        self.assertTrue(result.metrics["early_stopped"])
        self.assertEqual(result.metrics["num_iters"], 2)
        self.assertEqual(
            result.metrics["early_stop_reason"],
            "effect_success_cost_plateau",
        )

    def test_cost_plateau_does_not_stop_before_effect_success(self) -> None:
        model = mujoco.MjModel.from_xml_string(_XML)
        data = mujoco.MjData(model)
        config = PrimitiveCompilerConfig(
            num_rollouts=1,
            max_parallel_rollouts=1,
            num_nodes=2,
            num_elites=1,
            max_iters=5,
            min_iters=2,
            early_stop_patience=1,
            early_stop_cost_tolerance=1e-8,
            sigma_min=1e-6,
            sigma_max=1e-5,
            horizon=0.02,
            optimize_initial_pose=False,
        )

        result, _ = compile_primitive(
            PrimitiveCompileSpec(
                model=model,
                initial_state=np.concatenate((data.qpos, data.qvel)),
                nominal_knots=np.zeros((2, 1), dtype=float),
                sigma_init=np.zeros((2, 1), dtype=float),
                cost_fn=lambda states, controls: np.zeros(states.shape[0]),
                effect_predicate=lambda states, controls: np.zeros(
                    states.shape[0],
                    dtype=bool,
                ),
            ),
            config,
            verbose=False,
        )

        self.assertFalse(result.metrics["early_stopped"])
        self.assertEqual(result.metrics["num_iters"], 5)
        self.assertIsNone(result.metrics["early_stop_reason"])

    def test_mjwarp_batches_distinct_initial_states_as_problems(self) -> None:
        model = mujoco.MjModel.from_xml_string(_XML)
        original_density = float(model.opt.density)
        original_friction = model.geom_friction.copy()
        config = PrimitiveCompilerConfig(
            num_rollouts=4,
            max_parallel_rollouts=2,
            rollout_backend="mjwarp",
            rollout_device="cuda:1",
        )

        with mock.patch(
            "primitive_core.compile._load_mjwarp",
            return_value=(_FakeMJWarpBackend, _FakeWarp),
        ):
            backend, batch_size = _make_rollout_backend(model, config)

        self.assertEqual(batch_size, 4)
        self.assertEqual(backend.kwargs["num_threads"], 1)
        self.assertEqual(backend.kwargs["num_problems"], 4)
        self.assertEqual(backend.kwargs["device"], "cuda:1")

        warp_model = backend.kwargs["model"]
        self.assertEqual(float(warp_model.opt.density), 0.0)
        self.assertTrue(np.all(warp_model.geom_friction >= 0.01))
        self.assertEqual(float(model.opt.density), original_density)
        np.testing.assert_array_equal(model.geom_friction, original_friction)

    def test_mjwarp_request_fails_cleanly_without_cuda(self) -> None:
        model = mujoco.MjModel.from_xml_string(_XML)
        config = PrimitiveCompilerConfig(
            num_rollouts=4,
            rollout_backend="mjwarp",
        )
        unavailable_warp = mock.Mock()
        unavailable_warp.is_cuda_available.return_value = False

        with mock.patch(
            "primitive_core.compile._load_mjwarp",
            return_value=(_FakeMJWarpBackend, unavailable_warp),
        ):
            with self.assertRaisesRegex(RuntimeError, "CUDA is unavailable"):
                _make_rollout_backend(model, config)

    def test_cpu_rollout_chunking_preserves_candidate_initial_states(self) -> None:
        backend = _RecordingBackend()
        x0 = np.arange(10, dtype=np.float64).reshape(5, 2)
        controls = np.zeros((5, 3, 1), dtype=np.float64)

        states = _rollout_in_chunks(backend, x0, controls, chunk_size=2)

        np.testing.assert_array_equal(states[:, 0, :], x0)
        self.assertEqual([batch.shape[0] for batch in backend.x0_batches], [2, 2, 2])
        np.testing.assert_array_equal(backend.x0_batches[-1][0], x0[-1])
        np.testing.assert_array_equal(backend.x0_batches[-1][1], x0[-1])


if __name__ == "__main__":
    unittest.main()
