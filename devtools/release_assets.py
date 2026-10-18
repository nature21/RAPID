"""Package and verify non-Git inputs for the released simulation experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = "rapid-assets-manifest.json"
ASSET_ROOTS = (
    "assets/HOPE_object",
    "runtime_assets/benchmark_hope_v1",
    "tasks/sim_benchmark",
    "benchmark_private/sealed",
    "demos",
    "tasks/invention_real2sim",
    "runtime_assets/fr3",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_assets(root: Path) -> list[Path]:
    tracked = set(subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=root,
    ).decode().split("\0"))
    paths = []
    for relative in ASSET_ROOTS:
        directory = root / relative
        if not directory.is_dir():
            raise FileNotFoundError(f"Required asset directory is missing: {directory}")
        for path in sorted(directory.rglob("*")):
            name = path.relative_to(root)
            if (not path.is_file() or name.as_posix() in tracked
                    or "__pycache__" in name.parts
                    or path.suffix in {".pyc", ".pyo", ".zip"}):
                continue
            # Real-robot captures and alternative corpus candidates are not
            # inputs to the published simulation tasks.
            if relative == "tasks/invention_real2sim" and any(
                part.startswith("scene_") for part in name.parts[2:3]
            ):
                continue
            if path.is_symlink():
                raise ValueError(f"Asset must be a regular file: {path}")
            paths.append(path)
    return sorted(paths)


def build(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new archive path: {output}")
    paths = collect_assets(root)
    if not paths:
        raise ValueError("No non-Git assets found")
    manifest = {
        "schema": "rapid.release-assets/v1",
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
        ).strip(),
        "files": {
            p.relative_to(root).as_posix(): {
                "size": p.stat().st_size, "sha256": sha256(p),
            }
            for p in paths
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".zip.tmp")
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in paths:
                archive.write(path, path.relative_to(root).as_posix())
            archive.writestr(MANIFEST, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{sha256(output)}  {output.name}\n", encoding="utf-8",
    )
    return manifest


def verify(root: Path) -> int:
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    if manifest.get("schema") != "rapid.release-assets/v1":
        raise ValueError("Unsupported asset manifest")
    failures = []
    for name, record in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            raise ValueError(f"Asset path escapes repository: {name}")
        if (not path.is_file() or path.stat().st_size != record["size"]
                or sha256(path) != record["sha256"]):
            failures.append(name)
    if failures:
        raise ValueError("Missing or modified assets:\n" + "\n".join(failures))
    return len(manifest["files"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("build", help="Package local inputs; does not regenerate scenes")
    pack.add_argument("--output", type=Path, required=True)
    pack.add_argument("--root", type=Path, default=ROOT)
    check = sub.add_parser("verify", help="Verify assets after extracting the archive")
    check.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if args.command == "build":
        output = args.output.expanduser().resolve()
        manifest = build(root, output)
        print(f"Wrote {output}: {len(manifest['files'])} files, {output.stat().st_size} bytes")
        print(f"Checksum: {output}.sha256")
    else:
        print(f"Verified {verify(root)} asset files under {root}")


if __name__ == "__main__":
    main()
