import json
import subprocess
import zipfile

import pytest

from devtools.release_assets import ASSET_ROOTS, MANIFEST, build, sha256, verify


def test_asset_bundle_roundtrip_excludes_code_results_and_caches(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    for name in ASSET_ROOTS:
        (root / name).mkdir(parents=True, exist_ok=True)
    tracked = root / "demos/demo.mp4"
    tracked.write_bytes(b"tracked demonstration")
    subprocess.run(["git", "add", "demos/demo.mp4"], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                    "commit", "-qm", "Fixture"], cwd=root, check=True)
    asset = root / "runtime_assets/benchmark_hope_v1/mesh.obj"
    asset.write_text("v 0 0 0\n")
    cache = asset.parent / "__pycache__/module.pyc"
    cache.parent.mkdir(); cache.write_bytes(b"cache")
    old_zip = root / "assets/HOPE_object/source.zip"
    old_zip.write_bytes(b"redundant archive")
    result = root / "suite_runs/trial/private.txt"
    result.parent.mkdir(parents=True); result.write_text("private result")
    output = tmp_path / "assets.zip"
    manifest = build(root, output)
    assert set(manifest["files"]) == {asset.relative_to(root).as_posix()}
    assert output.with_suffix(".zip.sha256").read_text().split()[0] == sha256(output)
    restored = tmp_path / "checkout"
    with zipfile.ZipFile(output) as archive:
        archive.extractall(restored)
    assert verify(restored) == 1
    (restored / asset.relative_to(root)).write_text("changed")
    with pytest.raises(ValueError, match="Missing or modified"):
        verify(restored)
    with pytest.raises(FileExistsError):
        build(root, output)


def test_asset_manifest_rejects_paths_outside_checkout(tmp_path):
    (tmp_path / MANIFEST).write_text(json.dumps({
        "schema": "rapid.release-assets/v1",
        "files": {"../outside": {"size": 0, "sha256": ""}},
    }))
    with pytest.raises(ValueError, match="escapes repository"):
        verify(tmp_path)
