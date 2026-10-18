"""Host-only command line for constructing and auditing the sealed corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .assets import curate_hope_assets
from .certify import certify_case
from .generator import materialize_case, tree_sha256
from .specs import (
    BENCHMARK_VERSION,
    CASES_PER_STRATEGY,
    ROOT_SEED,
    STRATEGY_CONTRACTS,
    sampling_fingerprint,
    variants_for_strategy,
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def curated_asset_root(root: Path) -> Path:
    return root / "runtime_assets" / "benchmark_hope_v1"


def private_seal_root(root: Path) -> Path:
    return root / "benchmark_private" / "sealed" / BENCHMARK_VERSION


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_strategies(values: Sequence[int] | None) -> tuple[int, ...]:
    selected = tuple(values or sorted(STRATEGY_CONTRACTS))
    unknown = sorted(set(selected) - set(STRATEGY_CONTRACTS))
    if unknown:
        raise ValueError(f"unknown strategy IDs: {unknown}")
    return selected


def _cmd_curate(args: argparse.Namespace) -> int:
    root = repo_root()
    manifest = curate_hope_assets(
        args.raw_root or root / "assets" / "HOPE_object",
        args.output or curated_asset_root(root),
    )
    print(f"curated {len(manifest['assets'])} HOPE assets")
    print(args.output or curated_asset_root(root))
    return 0


def _case_manifest_path(seal_root: Path, strategy_id: int, case_index: int) -> Path:
    return seal_root / "manifests" / f"strategy_{strategy_id}" / f"case_{case_index:03d}.json"


def _cmd_generate(args: argparse.Namespace) -> int:
    root = repo_root()
    asset_root = args.asset_root or curated_asset_root(root)
    production_scene_root = root / "tasks" / "sim_benchmark"
    scene_root = args.scene_root or production_scene_root
    if not scene_root.is_absolute():
        scene_root = root / scene_root
    scene_root = scene_root.resolve()
    try:
        scene_root.relative_to(root / "tasks")
    except ValueError as exc:
        raise ValueError(
            f"generated benchmark scenes must live under {root / 'tasks'}"
        ) from exc

    seal_root = args.seal_root or private_seal_root(root)
    if not seal_root.is_absolute():
        seal_root = root / seal_root
    seal_root = seal_root.resolve()
    strategies = _selected_strategies(args.strategy)
    seal_path = seal_root / "seal.json"
    if seal_path.is_file():
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
    else:
        seal = {
            "benchmark_version": BENCHMARK_VERSION,
            "root_seed": ROOT_SEED,
            "sealed": False,
            "strategies": {},
        }

    for strategy_id in strategies:
        contract = STRATEGY_CONTRACTS[strategy_id]
        strategy_root = scene_root / contract.key
        records: list[dict[str, Any]] = []
        print(f"[benchmark] strategy {strategy_id}: generating {CASES_PER_STRATEGY} cases")
        for parameters in variants_for_strategy(strategy_id):
            destination = strategy_root / parameters.case_name
            if destination.exists() and not args.force:
                raise FileExistsError(
                    f"generated case already exists: {destination}; pass --force to replace scored cases"
                )
            staging_parent = strategy_root / ".generation_staging"
            staging_parent.mkdir(parents=True, exist_ok=True)
            staging = Path(
                tempfile.mkdtemp(prefix=f"{parameters.case_name}.", dir=staging_parent)
            )
            shutil.rmtree(staging)
            try:
                private_record = materialize_case(
                    root,
                    asset_root,
                    contract,
                    parameters,
                    destination=staging,
                )
                if args.no_certify:
                    certification = {
                        "passed": None,
                        "certifier": "skipped_by_cli",
                        "gates": [],
                    }
                else:
                    certification = certify_case(
                        staging,
                        contract,
                        private_record,
                        check_visibility=not args.skip_visibility,
                    ).to_json()
                private_record["certification"] = certification
                if certification["passed"] is False:
                    rejection = dict(private_record)
                    rejection["rejected"] = True
                    _write_json(
                        seal_root
                        / "rejections"
                        / f"strategy_{strategy_id}"
                        / f"case_{parameters.case_index:03d}.json",
                        rejection,
                    )
                    failures = [
                        gate["name"]
                        for gate in certification["gates"]
                        if not gate["passed"]
                    ]
                    raise RuntimeError(
                        f"case {strategy_id}/{parameters.case_index:03d} failed certification: {failures}"
                    )
                if destination.exists():
                    shutil.rmtree(destination)
                staging.rename(destination)
                private_record["case_sha256"] = tree_sha256(destination)
                manifest_path = _case_manifest_path(
                    seal_root, strategy_id, parameters.case_index
                )
                _write_json(manifest_path, private_record)
                records.append(
                    {
                        "case_index": parameters.case_index,
                        "case_sha256": private_record["case_sha256"],
                        "manifest_sha256": _file_sha256(manifest_path),
                        "certified": certification["passed"] is True,
                        "bucket": parameters.bucket,
                        "diagnostic_dimension": parameters.diagnostic_dimension,
                    }
                )
                print(
                    f"[benchmark] strategy {strategy_id} case {parameters.case_index:03d}: "
                    f"{parameters.bucket} certified"
                )
            finally:
                shutil.rmtree(staging, ignore_errors=True)
                if staging_parent.is_dir() and not any(staging_parent.iterdir()):
                    staging_parent.rmdir()

        seal["strategies"][str(strategy_id)] = {
            "sampling_sha256": sampling_fingerprint(strategy_id),
            "case_count": len(records),
            "cases": records,
        }
        seal["sealed"] = all(
            item.get("case_count") == CASES_PER_STRATEGY
            and all(case.get("certified") is True for case in item.get("cases", []))
            for item in seal["strategies"].values()
        ) and len(seal["strategies"]) == len(STRATEGY_CONTRACTS)
        _write_json(seal_path, seal)
    print(f"[benchmark] seal: {seal_path}")
    return 0


def _cmd_seal(args: argparse.Namespace) -> int:
    """Rebuild the private seal from already generated, certified artifacts."""

    root = repo_root()
    seal_root = args.seal_root or private_seal_root(root)
    strategies: dict[str, Any] = {}
    for strategy_id in sorted(STRATEGY_CONTRACTS):
        records: list[dict[str, Any]] = []
        for parameters in variants_for_strategy(strategy_id):
            case_dir = (
                root
                / "tasks"
                / "sim_benchmark"
                / f"strategy_{strategy_id}"
                / parameters.case_name
            )
            manifest_path = _case_manifest_path(
                seal_root, strategy_id, parameters.case_index
            )
            if not case_dir.is_dir() or not manifest_path.is_file():
                raise FileNotFoundError(
                    f"cannot seal missing case or manifest: {case_dir}, {manifest_path}"
                )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("certification", {}).get("passed") is not True:
                raise ValueError(
                    f"cannot seal uncertified case: strategy {strategy_id} "
                    f"case {parameters.case_index:03d}"
                )
            case_sha256 = tree_sha256(case_dir)
            if manifest.get("case_sha256") != case_sha256:
                raise ValueError(f"private manifest case hash mismatch: {manifest_path}")
            records.append(
                {
                    "case_index": parameters.case_index,
                    "case_sha256": case_sha256,
                    "manifest_sha256": _file_sha256(manifest_path),
                    "certified": True,
                    "bucket": parameters.bucket,
                    "diagnostic_dimension": parameters.diagnostic_dimension,
                }
            )
        strategies[str(strategy_id)] = {
            "sampling_sha256": sampling_fingerprint(strategy_id),
            "case_count": len(records),
            "cases": records,
        }
    seal = {
        "benchmark_version": BENCHMARK_VERSION,
        "root_seed": ROOT_SEED,
        "sealed": True,
        "strategies": strategies,
    }
    path = seal_root / "seal.json"
    _write_json(path, seal)
    case_count = sum(item["case_count"] for item in strategies.values())
    print(f"[benchmark] sealed {case_count} public cases and private manifests: {path}")
    return 0


def audit_corpus(root: Path, seal_root: Path) -> list[str]:
    findings: list[str] = []
    seal_path = seal_root / "seal.json"
    if not seal_path.is_file():
        return [f"missing seal: {seal_path}"]
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    if seal.get("benchmark_version") != BENCHMARK_VERSION:
        findings.append("seal benchmark version mismatch")
    if not seal.get("sealed"):
        findings.append("corpus is not marked sealed")
    seen_hashes: set[str] = set()
    for strategy_id in sorted(STRATEGY_CONTRACTS):
        record = seal.get("strategies", {}).get(str(strategy_id))
        if not isinstance(record, dict):
            findings.append(f"strategy {strategy_id}: missing seal entry")
            continue
        if record.get("sampling_sha256") != sampling_fingerprint(strategy_id):
            findings.append(f"strategy {strategy_id}: sampling fingerprint mismatch")
        cases = record.get("cases", [])
        if len(cases) != CASES_PER_STRATEGY:
            findings.append(f"strategy {strategy_id}: expected 50 sealed cases, found {len(cases)}")
        for case in cases:
            case_index = int(case["case_index"])
            if case.get("certified") is not True:
                findings.append(
                    f"strategy {strategy_id} case {case_index:03d}: not certified"
                )
            manifest_path = _case_manifest_path(seal_root, strategy_id, case_index)
            if not manifest_path.is_file():
                findings.append(
                    f"strategy {strategy_id} case {case_index:03d}: missing private manifest"
                )
            else:
                if _file_sha256(manifest_path) != case.get("manifest_sha256"):
                    findings.append(
                        f"strategy {strategy_id} case {case_index:03d}: "
                        "private manifest checksum mismatch"
                    )
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("certification", {}).get("passed") is not True:
                    findings.append(
                        f"strategy {strategy_id} case {case_index:03d}: "
                        "private certification did not pass"
                    )
            case_dir = (
                root
                / "tasks"
                / "sim_benchmark"
                / f"strategy_{strategy_id}"
                / f"case_{case_index:03d}_scene"
            )
            if not case_dir.is_dir():
                findings.append(f"strategy {strategy_id} case {case_index:03d}: missing package")
                continue
            actual = tree_sha256(case_dir)
            if actual != case.get("case_sha256"):
                findings.append(f"strategy {strategy_id} case {case_index:03d}: checksum mismatch")
            if actual in seen_hashes:
                findings.append(f"strategy {strategy_id} case {case_index:03d}: duplicate case fingerprint")
            seen_hashes.add(actual)
            public_text = "\n".join(
                path.read_text(encoding="utf-8", errors="ignore")
                for path in case_dir.rglob("*")
                if path.is_file() and path.suffix in {".py", ".json", ".txt", ".xml"}
            )
            for private_key in ("authoritative_roles", "diagnostic_dimension", "stress_axis", "mass_scale"):
                if private_key in public_text:
                    findings.append(
                        f"strategy {strategy_id} case {case_index:03d}: leaked private key {private_key}"
                    )
    return findings


def _cmd_audit(args: argparse.Namespace) -> int:
    root = repo_root()
    findings = audit_corpus(root, args.seal_root or private_seal_root(root))
    if findings:
        for finding in findings:
            print(f"[benchmark] FAIL: {finding}", file=sys.stderr)
        return 1
    case_count = len(STRATEGY_CONTRACTS) * CASES_PER_STRATEGY
    print(f"[benchmark] corpus audit passed: {case_count} unique sealed cases")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    curate = subparsers.add_parser("curate-assets")
    curate.add_argument("--raw-root", type=Path, default=None)
    curate.add_argument("--output", type=Path, default=None)
    curate.set_defaults(handler=_cmd_curate)

    generate = subparsers.add_parser("generate")
    generate.add_argument("--strategy", type=int, action="append", default=None)
    generate.add_argument("--asset-root", type=Path, default=None)
    generate.add_argument(
        "--scene-root",
        type=Path,
        default=None,
        help="Destination containing strategy_N directories; must live under tasks/.",
    )
    generate.add_argument("--seal-root", type=Path, default=None)
    generate.add_argument("--force", action="store_true")
    generate.add_argument("--no-certify", action="store_true")
    generate.add_argument(
        "--skip-visibility",
        action="store_true",
        help="Skip optional camera-visibility diagnostics.",
    )
    generate.set_defaults(handler=_cmd_generate)

    seal = subparsers.add_parser("seal")
    seal.add_argument("--seal-root", type=Path, default=None)
    seal.set_defaults(handler=_cmd_seal)

    audit = subparsers.add_parser("audit")
    audit.add_argument("--seal-root", type=Path, default=None)
    audit.set_defaults(handler=_cmd_audit)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (FileNotFoundError, FileExistsError, KeyError, RuntimeError, ValueError) as exc:
        print(f"[benchmark] error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
