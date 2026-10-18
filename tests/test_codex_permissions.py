from __future__ import annotations

import os
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.sandbox_validation import PERMISSION_PROFILE
from suites.staged.codex_launcher import (
    _fresh_codex_home,
    _instantiation_workspace_grants,
    _link_shared_codex_tmp,
    _parse_args,
    _safe_python_agent_args,
    _sync_gpu_render_client,
    _sync_instantiation_runtime,
    prepare_launch,
)
from suites.staged.codex_permissions import (
    PROFILE_NAME,
    codex_cli_runtime_read_grants,
    path_exposes_blocked_checkpoint,
    run_codex_helper_path,
    run_codex_home_path,
    write_codex_home_bootstrap,
    write_workspace_codex_config,
)
from suites.staged.workspaces import create_isolated_workspace


def _touch(path: Path, text: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_repo(root: Path) -> Path:
    repo = root / "repo"
    for dirname in (
        "workspace/src/primitive_core",
        "workspace/src/instantiation_core",
        "workspace/src/instantiations/primitives",
        "workspace/src/instantiations/strategies",
        "workspace/src/primitives",
        "workspace/src/scene_io",
        "workspace/src/strategies",
        "scripts",
        "agent",
        "agent/templates/invent_workspace/.agents/skills/rapid-construct-primitive",
        "agent/templates/invent_workspace/.agents/skills/rapid-compose-strategy",
        "agent/templates/invent_workspace/.agents/skills/rapid-instantiate-primitive",
        "agent/templates/invent_workspace/.agents/skills/rapid-instantiate-strategy",
        "agent/templates/invent_workspace/.agents/skills/rapid-deploy-strategy",
        "third_party/judo/judo",
        "runtime_assets/fr3/fr3_task_v1",
        "runtime_assets/benchmark_hope_v1",
        "benchmark_private",
        "tasks/invention_real2sim/other",
        "tasks/invention_real2sim/sample",
        "tasks/invention_real2sim/push",
        "tasks/invention_real2sim/demo_1_scene",
        "tasks/sim_benchmark/strategy_1/case_000_scene",
        "tasks/sim_benchmark/strategy_1/case_001_scene",
        "demos/construction",
        "demos/deploy",
        "suite_runs/old/workspace/src",
    ):
        (repo / dirname).mkdir(parents=True, exist_ok=True)
    _touch(repo / "scripts/rapid-python", "#!/usr/bin/env bash\n")
    _touch(repo / "workspace/src/scene_io/record.py", "# canonical recorder\n")
    _touch(
        repo / "workspace/src/instantiation_core/entrypoint.py",
        "# canonical entrypoint\n",
    )
    _touch(repo / "workspace/src/instantiation_core/runner.py", "# canonical runner\n")
    _touch(repo / "workspace/src/strategies/runtime.py", "# canonical strategy runtime\n")
    _touch(repo / "AGENTS.md", "# test\n")
    _touch(repo / "SUITE_RUNS.md", "# test\n")
    _touch(repo / "agent/templates/invent_workspace/AGENTS.md", "# invent\n")
    _touch(
        repo / "agent/templates/invent_workspace/.agents/skills/rapid-construct-primitive/SKILL.md",
        "# primitive skill\n",
    )
    _touch(
        repo / "agent/templates/invent_workspace/.agents/skills/rapid-compose-strategy/SKILL.md",
        "# strategy skill\n",
    )
    _touch(
        repo
        / "agent/templates/invent_workspace/.agents/skills/"
        "rapid-instantiate-primitive/SKILL.md",
        "# instantiate primitive skill\n",
    )
    _touch(
        repo
        / "agent/templates/invent_workspace/.agents/skills/"
        "rapid-instantiate-strategy/SKILL.md",
        "# instantiate strategy skill\n",
    )
    _touch(
        repo
        / "agent/templates/invent_workspace/.agents/skills/"
        "rapid-deploy-strategy/SKILL.md",
        "# deploy strategy skill\n",
    )
    _touch(repo / "tasks/invention_real2sim/other/scene.py", "")
    _touch(repo / "tasks/invention_real2sim/sample/scene.py", "")
    _touch(repo / "tasks/invention_real2sim/_scene_helpers.py", "HELPER = True\n")
    _touch(repo / "tasks/invention_real2sim/push/scene.py", "")
    _touch(repo / "tasks/invention_real2sim/demo_1_scene/scene.py", "")
    _touch(repo / "tasks/sim_benchmark/strategy_1/case_000_scene/scene.py", "")
    _touch(repo / "tasks/sim_benchmark/strategy_1/case_001_scene/scene.py", "")
    _touch(repo / "runtime_assets/fr3/fr3_task_v1/robot.xml", "<mujoco/>\n")
    _touch(repo / "demos/primitive/push/rgb.mp4", "demo")
    _touch(repo / "demos/strategy/demo_1/rgb.mp4", "demo")
    return repo


class CodexPermissionConfigTests(unittest.TestCase):
    def test_validation_worker_uses_generated_permission_profile(self) -> None:
        self.assertEqual(PERMISSION_PROFILE, PROFILE_NAME)

    def test_fresh_codex_home_links_only_the_shared_codex_tmp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = root / "repo"
            workspace = repo / "suite_runs/new/workspace"
            workspace.mkdir(parents=True)
            shared_tmp = root / "host-codex-home/.tmp"

            _, codex_home = _fresh_codex_home(
                workspace,
                repo_root=repo,
                codex_home_root=None,
                shared_codex_tmp=shared_tmp,
            )

            tmp_link = codex_home / ".tmp"
            self.assertTrue(tmp_link.is_symlink())
            self.assertEqual(tmp_link.resolve(), shared_tmp.resolve())
            self.assertTrue((codex_home / "tmp/arg0").is_dir())
            self.assertNotEqual(codex_home.resolve(), shared_tmp.parent.parent.resolve())

    def test_shared_codex_tmp_does_not_replace_existing_run_cache(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            codex_home = root / "run/codex_home"
            run_tmp = codex_home / ".tmp"
            shared_tmp = root / "shared/tmp"
            _touch(run_tmp / "marker", "run-local")

            linked = _link_shared_codex_tmp(codex_home, shared_tmp)

            self.assertFalse(linked)
            self.assertFalse(run_tmp.is_symlink())
            self.assertEqual((run_tmp / "marker").read_text(), "run-local")

    def test_gpu_render_client_refreshes_reused_workspace_infrastructure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            src = repo / "suite_runs/reused/workspace/src"
            _touch(src / "scene_io/record.py", "# stale recorder\n")

            changed = _sync_gpu_render_client(repo_root=repo, src=src.resolve())

            self.assertTrue(changed)
            self.assertEqual(
                (src / "scene_io/record.py").read_text(encoding="utf-8"),
                "# canonical recorder\n",
            )

    def test_instantiation_runtime_refreshes_reused_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            src = repo / "suite_runs/reused/workspace/src"
            _touch(src / "instantiation_core/entrypoint.py", "# stale\n")
            _touch(src / "instantiation_core/runner.py", "# stale\n")
            _touch(src / "strategies/runtime.py", "# stale\n")

            changed = _sync_instantiation_runtime(
                repo_root=repo,
                src=src.resolve(),
            )

            self.assertTrue(changed)
            self.assertEqual(
                (src / "instantiation_core/entrypoint.py").read_text(
                    encoding="utf-8"
                ),
                "# canonical entrypoint\n",
            )
            self.assertEqual(
                (src / "instantiation_core/runner.py").read_text(
                    encoding="utf-8"
                ),
                "# canonical runner\n",
            )
            self.assertEqual(
                (src / "strategies/runtime.py").read_text(encoding="utf-8"),
                "# canonical strategy runtime\n",
            )

    def test_shared_codex_tmp_skips_run_home_itself(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            codex_home = Path(tmp) / "run/codex_home"
            shared_tmp = codex_home / ".tmp"

            linked = _link_shared_codex_tmp(codex_home, shared_tmp)

            self.assertFalse(linked)
            self.assertFalse(shared_tmp.exists())

    def test_workspace_config_filters_broad_checkpoints_but_allows_selected_demo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            workspace = repo / "suite_runs/new/workspace"
            (workspace / "src").mkdir(parents=True)
            conda_root = Path(tmp) / "conda"
            conda_root.mkdir()
            selected_demo = repo / "demos/primitive/push/rgb.mp4"

            config_path = write_workspace_codex_config(
                workspace,
                repo_root=repo,
                read_grants=[repo / "tasks", repo / "suite_runs"],
                allowed_blocked_read_grants=[selected_demo],
                conda_root=conda_root,
            )

            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(data["default_permissions"], PROFILE_NAME)
            filesystem = data["permissions"][PROFILE_NAME]["filesystem"]
            self.assertEqual(filesystem[":root"], "deny")
            self.assertEqual(filesystem[":minimal"], "read")
            self.assertEqual(filesystem[":slash_tmp"], "deny")
            self.assertEqual(filesystem[str(selected_demo)], "read")
            self.assertNotIn(str(repo / "tasks"), filesystem)
            self.assertNotIn(str(repo / "suite_runs"), filesystem)
            self.assertNotIn(str(repo / "benchmark_private"), filesystem)
            self.assertNotIn(
                str(repo / "runtime_assets" / "benchmark_hope_v1"),
                filesystem,
            )
            fr3_asset = (repo / "runtime_assets/fr3/fr3_task_v1").resolve()
            self.assertEqual(filesystem[str(fr3_asset)], "read")
            self.assertNotIn(str(fr3_asset.parent), filesystem)

            workspace_roots = filesystem[":workspace_roots"]
            self.assertEqual(workspace_roots["."], "write")
            self.assertEqual(workspace_roots[".codex"], "read")
            self.assertEqual(workspace_roots["task_inputs"], "read")
            self.assertEqual(workspace_roots["agent_work/codex_home"], "deny")
            self.assertEqual(workspace_roots["**/*.env"], "deny")
            self.assertEqual(filesystem[str(run_codex_helper_path(workspace))], "read")
            self.assertTrue(run_codex_helper_path(workspace).is_dir())

    def test_bootstrap_trusts_only_exact_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "run/workspace"
            codex_home = root / "codex-home"
            workspace.mkdir(parents=True)

            config_path = write_codex_home_bootstrap(codex_home, workspace=workspace)
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(data["project_root_markers"], [])
            self.assertEqual(
                data["projects"][str(workspace.resolve())]["trust_level"],
                "trusted",
            )

    def test_bootstrap_copies_provider_defaults_but_not_legacy_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "run/workspace"
            codex_home = root / "codex-home"
            host_config = root / "host-config.toml"
            workspace.mkdir(parents=True)
            host_config.write_text(
                "\n".join(
                    [
                        'model_provider = "crs"',
                        'preferred_auth_method = "apikey"',
                        'sandbox_mode = "workspace-write"',
                        'default_permissions = ":workspace"',
                        "",
                        "[model_providers.crs]",
                        'name = "crs"',
                        'base_url = "https://example.invalid/openai"',
                        'wire_api = "responses"',
                        "requires_openai_auth = true",
                    ]
                ),
                encoding="utf-8",
            )

            config_path = write_codex_home_bootstrap(
                codex_home,
                workspace=workspace,
                host_config=host_config,
            )
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))

            self.assertEqual(data["model_provider"], "crs")
            self.assertEqual(data["preferred_auth_method"], "apikey")
            self.assertTrue(data["model_providers"]["crs"]["requires_openai_auth"])
            self.assertNotIn("sandbox_mode", data)
            self.assertNotIn("default_permissions", data)

    def test_codex_cli_runtime_grants_detect_npm_package_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            package_root = root / "node/lib/node_modules/@openai/codex"
            bin_dir = root / "node/bin"
            codex_js = package_root / "bin/codex.js"
            codex_link = bin_dir / "codex"
            _touch(codex_js, "#!/usr/bin/env node\n")
            codex_js.chmod(0o755)
            bin_dir.mkdir(parents=True)
            codex_link.symlink_to(codex_js)

            with patch.dict(os.environ, {"PATH": str(bin_dir)}, clear=False):
                grants = codex_cli_runtime_read_grants()

            self.assertIn(package_root.resolve(), grants)

    def test_codex_cli_runtime_grants_include_npm_platform_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            packages = Path(tmp) / "node_modules/@openai"
            launcher = packages / "codex/bin/codex.js"
            native = packages / "codex-linux-x64"
            unrelated = packages / "other-tool"
            _touch(launcher, "#!/usr/bin/env node\n")
            _touch(native / "vendor/x86_64-unknown-linux-musl/bin/codex")
            _touch(unrelated / "vendor/tool")
            with patch("shutil.which", return_value=str(launcher)), patch.dict(
                os.environ, {}, clear=True
            ):
                grants = codex_cli_runtime_read_grants()
            self.assertEqual(set(grants), {launcher.parent.parent, native})

    def test_create_workspace_writes_workspace_codex_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            workspace_info = create_isolated_workspace(
                repo / "suite_runs/new",
                repo_root=repo,
            )

            config_path = workspace_info.workspace / ".codex/config.toml"
            self.assertTrue(config_path.is_file())
            self.assertEqual(
                (workspace_info.workspace / "AGENTS.md").read_text(encoding="utf-8"),
                "# invent\n",
            )
            self.assertTrue(
                (
                    workspace_info.workspace
                    / ".agents/skills/rapid-construct-primitive/SKILL.md"
                ).is_file()
            )
            self.assertTrue(
                (
                    workspace_info.workspace
                    / ".agents/skills/rapid-deploy-strategy/SKILL.md"
                ).is_file()
            )
            self.assertTrue(
                (
                    workspace_info.workspace
                    / ".agents/skills/rapid-compose-strategy/SKILL.md"
                ).is_file()
            )
            self.assertTrue(
                (
                    workspace_info.workspace
                    / ".agents/skills/rapid-instantiate-primitive/SKILL.md"
                ).is_file()
            )
            self.assertTrue(
                (
                    workspace_info.workspace
                    / ".agents/skills/rapid-instantiate-strategy/SKILL.md"
                ).is_file()
            )
            data = tomllib.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(data["default_permissions"], PROFILE_NAME)
            bootstrap_path = run_codex_home_path(workspace_info.workspace) / "config.toml"
            self.assertTrue(bootstrap_path.is_file())
            bootstrap = tomllib.loads(bootstrap_path.read_text(encoding="utf-8"))
            self.assertEqual(
                bootstrap["projects"][str(workspace_info.workspace.resolve())]["trust_level"],
                "trusted",
            )

    def test_prepare_launch_copies_task_inputs_and_rejects_broad_ro_bind(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            workspace = repo / "suite_runs/new/workspace"
            (workspace / "src").mkdir(parents=True)
            args = _parse_args(
                [
                    "--workspace",
                    str(workspace),
                    "--no-codex-auth",
                    "--",
                    "agent/construct_primitive_exec.py",
                    "demos/primitive/push/rgb.mp4",
                    "--env-module",
                    "tasks.invention_real2sim.sample",
                    "--name",
                    "sample",
                ]
            )
            thread_env = {
                variable: ""
                for variable in (
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                    "BLIS_NUM_THREADS",
                )
            }
            with patch.dict(
                os.environ,
                {"RAPID_REPO_ROOT": str(repo), **thread_env},
                clear=False,
            ):
                prepared = prepare_launch(args)

            copied_task = (
                workspace
                / "task_inputs/tasks/invention_real2sim/sample/scene.py"
            )
            self.assertTrue(copied_task.is_file())
            self.assertTrue(
                (
                    workspace
                    / "task_inputs/tasks/invention_real2sim/_scene_helpers.py"
                ).is_file()
            )
            self.assertFalse(
                (
                    workspace
                    / "task_inputs/tasks/invention_real2sim/other"
                ).exists()
            )
            self.assertTrue(prepared.codex_home_config.is_file())
            self.assertEqual(prepared.codex_home, run_codex_home_path(workspace))
            self.assertEqual(prepared.agent_home, (workspace / "agent_work/home").resolve())
            self.assertEqual(
                prepared.env["TMPDIR"],
                str((workspace / "agent_work/tmp").resolve()),
            )
            self.assertEqual(
                prepared.env["XDG_CACHE_HOME"],
                str((workspace / "agent_work/tmp/rapid_cache").resolve()),
            )
            self.assertEqual(
                prepared.env["MPLCONFIGDIR"],
                str(
                    (
                        workspace
                        / "agent_work/tmp/rapid_cache/matplotlib"
                    ).resolve()
                ),
            )
            for variable in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "BLIS_NUM_THREADS",
            ):
                self.assertEqual(prepared.env[variable], "1")
            self.assertEqual(
                prepared.command[:4],
                [
                    str(repo / "scripts/rapid-python"),
                    "-P",
                    "-m",
                    "agent.construct_primitive_exec",
                ],
            )
            workspace_config_text = prepared.workspace_config.read_text(encoding="utf-8")
            self.assertIn(str((repo / "demos/primitive/push/rgb.mp4").resolve()), workspace_config_text)
            filesystem = tomllib.loads(workspace_config_text)["permissions"][PROFILE_NAME][
                "filesystem"
            ]
            helper_dir = run_codex_helper_path(workspace)
            self.assertTrue(helper_dir.is_dir())
            self.assertEqual(filesystem[str(helper_dir.resolve())], "read")
            self.assertNotIn(str(prepared.codex_home.resolve()), filesystem)
            self.assertTrue(
                path_exposes_blocked_checkpoint(
                    repo / "tasks",
                    repo_root=repo,
                    workspace=workspace,
                )
            )

            codex_runs = Path(tmp) / "codex-runs"
            bad_args = _parse_args(
                [
                    "--workspace",
                    str(workspace),
                    "--codex-home-root",
                    str(codex_runs),
                    "--no-codex-auth",
                    "--ro-bind",
                    str(repo / "tasks"),
                    "--",
                    "agent/construct_primitive_exec.py",
                    "demos/primitive/push/rgb.mp4",
                ]
            )
            with patch.dict(os.environ, {"RAPID_REPO_ROOT": str(repo)}, clear=False):
                with self.assertRaises(SystemExit):
                    prepare_launch(bad_args)

    def test_prepare_launch_copies_assigned_scene_for_composition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            workspace = repo / "suite_runs/new/workspace"
            (workspace / "src").mkdir(parents=True)
            args = _parse_args(
                [
                    "--workspace",
                    str(workspace),
                    "--no-codex-auth",
                    "--",
                    "agent/compose_strategy_exec.py",
                    "demos/strategy/demo_1/rgb.mp4",
                    "--env-module",
                    "tasks.invention_real2sim.demo_1_scene",
                    "--name",
                    "strategy_1",
                ]
            )

            with patch.dict(os.environ, {"RAPID_REPO_ROOT": str(repo)}, clear=False):
                prepared = prepare_launch(args)

            self.assertTrue(
                (
                    workspace
                    / "task_inputs/tasks/invention_real2sim/demo_1_scene/scene.py"
                ).is_file()
            )
            self.assertFalse(
                (workspace / "task_inputs/tasks/invention_real2sim/push").exists()
            )
            self.assertTrue(
                (
                    workspace / "task_inputs/tasks/invention_real2sim/_scene_helpers.py"
                ).is_file()
            )
            data = tomllib.loads(
                prepared.workspace_config.read_text(encoding="utf-8")
            )
            roots = data["permissions"][PROFILE_NAME]["filesystem"][
                ":workspace_roots"
            ]
            self.assertEqual(roots["."], "write")

    def test_reused_benchmark_workspace_keeps_cases_without_copying_fr3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            workspace = repo / "suite_runs/new/workspace"
            (workspace / "src").mkdir(parents=True)

            for case in ("case_000_scene", "case_001_scene"):
                args = _parse_args(
                    [
                        "--workspace",
                        str(workspace),
                        "--no-codex-auth",
                        "--",
                        "agent/deploy_strategy_exec.py",
                        "--strategy",
                        "strategy_1",
                        "--env-module",
                        f"tasks.sim_benchmark.strategy_1.{case}",
                        "--name",
                        case,
                        "--goal",
                        "Complete the benchmark task.",
                    ]
                )
                with patch.dict(
                    os.environ,
                    {"RAPID_REPO_ROOT": str(repo)},
                    clear=False,
                ):
                    prepared = prepare_launch(args)

            staged_strategy = (
                workspace / "task_inputs/tasks/sim_benchmark/strategy_1"
            )
            self.assertTrue((staged_strategy / "case_000_scene/scene.py").is_file())
            self.assertTrue((staged_strategy / "case_001_scene/scene.py").is_file())
            self.assertEqual(list((workspace / "task_inputs").rglob("fr3")), [])

            filesystem = tomllib.loads(
                prepared.workspace_config.read_text(encoding="utf-8")
            )["permissions"][PROFILE_NAME]["filesystem"]
            fr3_asset = (repo / "runtime_assets/fr3/fr3_task_v1").resolve()
            self.assertEqual(filesystem[str(fr3_asset)], "read")
            self.assertNotIn(str(fr3_asset.parent), filesystem)

    def test_primitive_instantiation_only_grants_binding_package_write(self) -> None:
        src = Path("/workspace/src")

        grants, protected = _instantiation_workspace_grants(
            [
                "agent/instantiate_primitive_exec.py",
                "--primitive",
                "push",
                "--name",
                "renamed_scene",
            ],
            src=src,
        )

        self.assertEqual(
            grants,
            [src / "instantiations/primitives/push/renamed_scene"],
        )
        self.assertEqual(
            protected,
            [
                src / "instantiations/primitives/push/renamed_scene/__init__.py",
            ],
        )

    def test_strategy_deployment_grants_no_source_write(self) -> None:
        grants, protected = _instantiation_workspace_grants(
            [
                "agent/deploy_strategy_exec.py",
                "--strategy",
                "strategy_1",
                "--name",
                "renamed_scene",
            ],
            src=Path("/workspace/src"),
        )

        self.assertEqual(grants, [])
        self.assertEqual(protected, [])

    def test_strategy_deployment_uses_safe_module_execution(self) -> None:
        args = _safe_python_agent_args(
            ["agent/deploy_strategy_exec.py", "--strategy", "strategy_1"],
            repo_root=Path("/repo"),
        )

        self.assertEqual(
            args,
            [
                "-P",
                "-m",
                "agent.deploy_strategy_exec",
                "--strategy",
                "strategy_1",
            ],
        )

    def test_prepare_launch_copies_scene_for_instantiation_helper(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            workspace = repo / "suite_runs/new/workspace"
            (workspace / "src").mkdir(parents=True)
            args = _parse_args(
                [
                    "--workspace",
                    str(workspace),
                    "--no-codex-auth",
                    "--",
                    "agent/instantiate_strategy_exec.py",
                    "--strategy",
                    "strategy_1",
                    "--env-module",
                    "tasks.invention_real2sim.demo_1_scene",
                    "--name",
                    "renamed_scene",
                    "--goal",
                    "Store the target.",
                ]
            )

            with patch.dict(
                os.environ,
                {"RAPID_REPO_ROOT": str(repo)},
                clear=False,
            ):
                prepared = prepare_launch(args)

            self.assertTrue(
                (
                    workspace
                    / "task_inputs/tasks/invention_real2sim/"
                    "demo_1_scene/scene.py"
                ).is_file()
            )
            data = tomllib.loads(
                prepared.workspace_config.read_text(encoding="utf-8")
            )
            roots = data["permissions"][PROFILE_NAME]["filesystem"][
                ":workspace_roots"
            ]
            self.assertEqual(roots["."], "read")
            self.assertEqual(roots[".codex"], "read")
            self.assertEqual(roots["agent_work"], "write")
            self.assertEqual(roots["results"], "write")
            self.assertEqual(roots["results_instantiation"], "write")
            self.assertEqual(
                roots[
                    "src/instantiations/strategies/"
                    "strategy_1/renamed_scene"
                ],
                "write",
            )
            self.assertEqual(
                roots[
                    "src/instantiations/strategies/"
                    "strategy_1/renamed_scene/__init__.py"
                ],
                "read",
            )
            self.assertNotIn(
                "src/instantiations/strategies/"
                "strategy_1/renamed_scene/main.py",
                roots,
            )
            self.assertNotIn(
                "src/instantiations/strategies/"
                "strategy_1/renamed_scene/binding.json",
                roots,
            )
            self.assertNotIn("src/primitives", roots)
            self.assertNotIn("src/strategies", roots)
            self.assertNotIn("src/primitive_core", roots)
            self.assertNotIn("src/instantiation_core", roots)

    def test_prepare_launch_ignores_option_values_when_granting_demo_reads(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            workspace = repo / "suite_runs/new/workspace"
            (workspace / "src").mkdir(parents=True)
            long_description = (
                "The assigned scene uses the complete FR3 arm with Cartesian OSC. "
                "When performing nonprehensile motions with the gripper, "
                "make contact with the pusher_site, coincident with grasp_site, "
                "and better keep the gripper closed. The optimizer action order is "
                "x, y, z, roll, pitch, yaw, gripper."
            )
            args = _parse_args(
                [
                    "--workspace",
                    str(workspace),
                    "--no-codex-auth",
                    "--",
                    "agent/construct_primitive_exec.py",
                    "demos/primitive/push/rgb.mp4",
                    "--env-module",
                    "tasks.invention_real2sim.sample",
                    "--name",
                    "sample",
                    "--description",
                    long_description,
                ]
            )

            with patch.dict(os.environ, {"RAPID_REPO_ROOT": str(repo)}, clear=False):
                prepared = prepare_launch(args)

            filesystem = tomllib.loads(
                prepared.workspace_config.read_text(encoding="utf-8")
            )["permissions"][PROFILE_NAME]["filesystem"]
            self.assertIn(str((repo / "demos/primitive/push/rgb.mp4").resolve()), filesystem)
            self.assertFalse(any(long_description in path for path in filesystem))

    def test_external_codex_home_root_adds_exact_helper_grant(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = _make_repo(Path(tmp))
            workspace = repo / "suite_runs/new/workspace"
            (workspace / "src").mkdir(parents=True)
            codex_runs = Path(tmp) / "codex-runs"
            args = _parse_args(
                [
                    "--workspace",
                    str(workspace),
                    "--codex-home-root",
                    str(codex_runs),
                    "--no-codex-auth",
                    "--probe-permissions",
                ]
            )
            with patch.dict(os.environ, {"RAPID_REPO_ROOT": str(repo)}, clear=False):
                prepared = prepare_launch(args)

            filesystem = tomllib.loads(
                prepared.workspace_config.read_text(encoding="utf-8")
            )["permissions"][PROFILE_NAME]["filesystem"]
            helper_dir = prepared.codex_home / "tmp" / "arg0"
            self.assertEqual(prepared.codex_home, codex_runs.resolve() / "new_codex")
            self.assertEqual(filesystem[str(helper_dir.resolve())], "read")


if __name__ == "__main__":
    unittest.main()
