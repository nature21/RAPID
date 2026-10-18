from __future__ import annotations

import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
INVENT_TEMPLATE = REPO_ROOT / "agent/templates/invent_workspace"


class InventionGuidanceTests(unittest.TestCase):
    def test_invention_skills_define_budget_ownership_and_safe_early_stop(self) -> None:
        primitive = (
            INVENT_TEMPLATE
            / ".agents/skills/rapid-construct-primitive/SKILL.md"
        ).read_text(encoding="utf-8")
        strategy = (
            INVENT_TEMPLATE
            / ".agents/skills/rapid-compose-strategy/SKILL.md"
        ).read_text(encoding="utf-8")

        self.assertIn("REFERENCE_OPTIMIZATION_BUDGET", primitive)
        self.assertIn("elite_fraction", primitive)
        self.assertIn("predicate is false must run", primitive)
        self.assertIn("transit_seed=cfg.transit_seed", primitive)
        self.assertIn("PrimitivePhase.optimization_budget", strategy)
        self.assertIn("GLOBAL_MAX_OPTIMIZATION_BUDGET", strategy)
        self.assertIn("requested_budget", strategy)
        self.assertIn("predicate is false must run", strategy)
        self.assertIn("transit_seed=args.transit_seed", strategy)
        self.assertIn(
            "do not turn an unsatisfied primitive effect into a strategy exception",
            " ".join(strategy.split()),
        )
        self.assertIn("final `Psi` alone supplies", strategy)
        self.assertIn("primitive_phase_effect_success", strategy)

    def test_shared_guidance_removes_incidental_variant_objects(self) -> None:
        guidance = (INVENT_TEMPLATE / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn("only the causal role closure", guidance)
        self.assertIn("Remove incidental scene bodies", guidance)
        self.assertIn("does not prove task infeasibility", guidance)
        self.assertIn("exact runtime", guidance)

    def test_shared_guidance_excludes_workflow_specific_procedures(self) -> None:
        guidance = (INVENT_TEMPLATE / "AGENTS.md").read_text(encoding="utf-8")

        self.assertIn("Strategy contracts and bindings are role-only", guidance)
        self.assertIn("goal or derived subgoal", guidance)
        self.assertIn("declared mechanism or a measured phase effect", guidance)
        for workflow_specific_text in (
            "open-library strategy invention runs",
            "`results_instantiation/`",
            "32x6 minimum",
            "phase-wise partial progress `P(t)`",
            "Population costs must be batched",
            "Use `$rapid-construct-primitive`",
        ):
            with self.subTest(text=workflow_specific_text):
                self.assertNotIn(workflow_specific_text, guidance)

    def test_orientation_guidance_is_object_centric_and_stable(self) -> None:
        shared = (INVENT_TEMPLATE / "AGENTS.md").read_text(encoding="utf-8")
        primitive = (
            INVENT_TEMPLATE
            / ".agents/skills/rapid-construct-primitive/SKILL.md"
        ).read_text(encoding="utf-8")
        strategy = (
            INVENT_TEMPLATE
            / ".agents/skills/rapid-compose-strategy/SKILL.md"
        ).read_text(encoding="utf-8")
        shared_normalized = " ".join(shared.split())
        primitive_normalized = " ".join(primitive.split())
        strategy_normalized = " ".join(strategy.split())

        self.assertIn("object-centric", shared)
        self.assertIn(
            "do not supply an FR3 end-effector attitude",
            shared_normalized,
        )
        self.assertIn("local X is the short-thickness axis", shared_normalized)
        self.assertIn("local Y is the jaw-closing axis", shared_normalized)
        self.assertIn("world-space hand local-X axis", shared_normalized)
        self.assertIn("nearly parallel or antiparallel", shared_normalized)
        self.assertIn("unexplained fixed cant angle", shared_normalized)
        self.assertIn("stable phase-level attitude", shared)
        self.assertIn(
            "hand-local-X / support-normal posture rule",
            primitive_normalized,
        )
        self.assertIn("does not provide an FR3 attitude", primitive)
        self.assertIn("after free-space clearance and before `Approach`", primitive)
        self.assertIn("owns its task-required contact axes", strategy_normalized)
        self.assertIn("not an arbitrary full quaternion", strategy_normalized)
        self.assertIn(
            "`Reorient.equivalent_quaternion_offsets`",
            strategy_normalized,
        )
        self.assertIn("co-rotate every task-required axis", strategy_normalized)
        self.assertNotIn("gripper lateral axis", shared)
        self.assertNotIn(
            "Use `Reorient` only when the inherited attitude cannot", primitive
        )

    def test_both_invention_skills_distinguish_search_failure_from_infeasibility(self) -> None:
        for skill_name in ("rapid-construct-primitive", "rapid-compose-strategy"):
            with self.subTest(skill=skill_name):
                guidance = (
                    INVENT_TEMPLATE
                    / ".agents/skills"
                    / skill_name
                    / "SKILL.md"
                ).read_text(encoding="utf-8")
                normalized = " ".join(guidance.split())

                self.assertIn("remove bodies", guidance)
                self.assertIn("exact runtime scene", normalized)
                self.assertIn("unresolved invocations may enter", normalized)
                self.assertIn("necessary physical condition", normalized)
                self.assertIn("`scene_or_goal_infeasible`", guidance)
                self.assertIn("only when a retained", guidance)
                self.assertIn("compile the resulting `MjSpec`", guidance)
                self.assertIn("may not", guidance)
                self.assertNotIn("pass the invocation-feasibility gate", normalized)

    def test_both_invention_skills_bound_visual_feedback_checkpoints(self) -> None:
        for skill_name in ("rapid-construct-primitive", "rapid-compose-strategy"):
            with self.subTest(skill=skill_name):
                guidance = (
                    INVENT_TEMPLATE
                    / ".agents/skills"
                    / skill_name
                    / "SKILL.md"
                ).read_text(encoding="utf-8")
                normalized = " ".join(guidance.split())

                self.assertIn(
                    "visual feedback as a bounded debugging checkpoint", normalized
                )
                self.assertIn(
                    "one diagnostic render per design fingerprint", normalized
                )
                self.assertIn(
                    "single most informative unresolved example", normalized
                )
                self.assertIn("before further source tuning", normalized)
                self.assertIn("`--no-record`", guidance)
                self.assertIn("same physical failure signature", normalized)
                self.assertIn("distinct failure signature", normalized)
                self.assertIn("metrics/recording disagreement", normalized)
                self.assertIn("threshold-only change", normalized)
                self.assertIn("video path", normalized)

        strategy = (
            INVENT_TEMPLATE
            / ".agents/skills/rapid-compose-strategy/SKILL.md"
        ).read_text(encoding="utf-8")
        strategy_normalized = " ".join(strategy.split())
        self.assertIn(
            "shortest replay that preserves the failure context",
            strategy_normalized,
        )
        self.assertIn("phase-local primitive replay", strategy_normalized)
        self.assertIn("full-strategy replay", strategy_normalized)

    def test_both_invention_skills_require_evidence_for_predicate_calibration(self) -> None:
        for skill_name in ("rapid-construct-primitive", "rapid-compose-strategy"):
            with self.subTest(skill=skill_name):
                guidance = (
                    INVENT_TEMPLATE
                    / ".agents/skills"
                    / skill_name
                    / "SKILL.md"
                ).read_text(encoding="utf-8")
                normalized = " ".join(guidance.split())

                self.assertIn("replayable physical evidence", normalized)
                self.assertIn("false positive", normalized)
                self.assertIn("false negative", normalized)
                self.assertIn(
                    "Search difficulty or budget exhaustion does not justify "
                    "lowering a threshold or deleting a conjunct",
                    normalized,
                )
                self.assertIn("current and proposed formulations", normalized)
                self.assertIn("record every conjunct result", normalized)
                self.assertIn("goal text, parameter names and meanings", normalized)
                self.assertIn("fixed physical dwell or settle interval", normalized)
                self.assertIn("fraction of the optimized horizon", normalized)
                self.assertIn("source-fingerprint change", normalized)
                self.assertIn("bounded visual checkpoint", normalized)

        primitive = (
            INVENT_TEMPLATE
            / ".agents/skills/rapid-construct-primitive/SKILL.md"
        ).read_text(encoding="utf-8")
        strategy = (
            INVENT_TEMPLATE
            / ".agents/skills/rapid-compose-strategy/SKILL.md"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "Evaluate any required direction, axis, or reference-frame relation "
            "on the effect itself",
            " ".join(primitive.split()),
        )
        self.assertIn(
            "follow the primitive calibration protocol",
            " ".join(strategy.split()),
        )

    def test_both_invention_skills_treat_nominal_as_initialization(self) -> None:
        for skill_name in ("rapid-construct-primitive", "rapid-compose-strategy"):
            with self.subTest(skill=skill_name):
                guidance = (
                    INVENT_TEMPLATE
                    / ".agents/skills"
                    / skill_name
                    / "SKILL.md"
                ).read_text(encoding="utf-8")
                normalized = " ".join(guidance.split())

                self.assertIn(
                    "as an initialization, not as a candidate", normalized
                )
                self.assertIn(
                    "Nominal success, contact, and task progress are not "
                    "prerequisites",
                    normalized,
                )
                self.assertIn(
                    "Do not infer search support or edit learned motion",
                    normalized,
                )
                self.assertIn(
                    "After wiring passes, immediately run the minimum search",
                    normalized,
                )
                self.assertIn(
                    "then run the minimum search even when nominal",
                    normalized,
                )
                self.assertIn(
                    "Nominal failure alone is not such evidence", normalized
                )

    def test_both_invention_skills_reserve_instantiation_results(self) -> None:
        for skill_name in ("rapid-construct-primitive", "rapid-compose-strategy"):
            with self.subTest(skill=skill_name):
                guidance = (
                    INVENT_TEMPLATE
                    / ".agents/skills"
                    / skill_name
                    / "SKILL.md"
                ).read_text(encoding="utf-8")
                normalized = " ".join(guidance.split())

                self.assertIn(
                    "Keep `results_instantiation/` empty throughout invention",
                    normalized,
                )
                self.assertIn(
                    "reserved exclusively for later deploy-instantiation runs",
                    normalized,
                )
                self.assertIn("Every invention-time execution must pass", normalized)
                self.assertIn("`agent_work/debug/`", normalized)
                self.assertIn("`results/` only for retained evidence", normalized)

    def test_both_invention_skills_act_on_safe_handoff_projection(self) -> None:
        for skill_name in ("rapid-construct-primitive", "rapid-compose-strategy"):
            with self.subTest(skill=skill_name):
                guidance = (
                    INVENT_TEMPLATE
                    / ".agents/skills"
                    / skill_name
                    / "SKILL.md"
                ).read_text(encoding="utf-8")
                normalized = " ".join(guidance.split())

                self.assertIn(
                    "`next_primitive_initial_knot_requires_safe_handoff_projection`",
                    normalized,
                )
                self.assertIn("`initial_knot_advisory`", normalized)
                self.assertIn("collision-free precontact", normalized)
                self.assertIn("later contact-rich trajectory", normalized)

    def test_both_invention_skills_describe_current_transit_runtime(self) -> None:
        for skill_name in ("rapid-construct-primitive", "rapid-compose-strategy"):
            with self.subTest(skill=skill_name):
                guidance = (
                    INVENT_TEMPLATE
                    / ".agents/skills"
                    / skill_name
                    / "SKILL.md"
                ).read_text(encoding="utf-8")
                normalized = " ".join(guidance.split())

                self.assertIn("fixed-aperture arm motion", normalized)
                self.assertIn("neither opens the gripper nor searches", normalized)
                self.assertIn("continuous local SE(3) region", normalized)
                self.assertIn("10 mm path clearance", normalized)
                self.assertIn("3 mm tracking reserve", normalized)
                self.assertIn("at most four diverse native-OSC candidates", normalized)
                self.assertIn(
                    "first measured violating pair, step, and normal",
                    normalized,
                )
                self.assertIn("may not cross into contact", normalized)
                self.assertIn("full 0.100 m ball", normalized)
                self.assertIn("2 mm radial resolution", normalized)
                self.assertIn("selected attitude and aperture fixed", normalized)
                self.assertIn("declared equivalents for IK first", normalized)
                self.assertIn("not restricted to fixed rays", normalized)
                self.assertIn("structured advisories", normalized)
                self.assertIn("do not stop later", normalized)
                self.assertIn("complete beginning-to-failure", normalized)
                self.assertIn(
                    "kinematic planning or release failure remains hard",
                    normalized,
                )
                self.assertNotIn("runtime-owned opening command", normalized)
                self.assertNotIn("releases any prior grasp", normalized)

    def test_invention_guidance_uses_capability_bounds_and_terminal_probes(self) -> None:
        shared = (INVENT_TEMPLATE / "AGENTS.md").read_text(encoding="utf-8")
        primitive = (
            INVENT_TEMPLATE
            / ".agents/skills/rapid-construct-primitive/SKILL.md"
        ).read_text(encoding="utf-8")
        strategy = (
            INVENT_TEMPLATE
            / ".agents/skills/rapid-compose-strategy/SKILL.md"
        ).read_text(encoding="utf-8")

        shared_normalized = " ".join(shared.split())
        primitive_normalized = " ".join(primitive.split())
        strategy_normalized = " ".join(strategy.split())

        self.assertIn("arbitrary absolute cutoff", shared_normalized)
        self.assertIn("bounded decomposition", shared_normalized)
        self.assertIn("may not create the postcondition", shared_normalized)
        self.assertIn("run the probe only for retained candidates", shared_normalized)
        self.assertIn("release feasibility separately", shared_normalized)
        self.assertIn("reference search scale is not such a bound", primitive_normalized)
        self.assertIn("semantically invalid inputs", primitive_normalized)
        self.assertIn("bounded terminal-stability probe", primitive_normalized)
        self.assertIn("may not introduce additional hard cutoffs", strategy_normalized)
        self.assertIn("bounded decomposition", strategy_normalized)
        self.assertIn("bounded terminal-stability probe", strategy_normalized)


if __name__ == "__main__":
    unittest.main()
