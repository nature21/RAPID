"""Track editable scene assets across agent repair and task packaging."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET


def scene_asset_paths(directory: Path) -> set[Path]:
    """Resolve MJCF assets and OBJ material/texture dependencies within a bundle."""

    from real2sim.package_scene import _collect_referenced_assets

    directory = directory.resolve()
    root = ET.parse(directory / "scene.xml").getroot()
    assets = _collect_referenced_assets(directory, root)
    paths = set()
    for path in assets:
        resolved = path.resolve()
        if not resolved.is_relative_to(directory):
            raise ValueError(f"scene asset escapes bundle: {path}")
        relative = resolved.relative_to(directory)
        if relative.parts[0] == "agent_repair":
            raise ValueError("runtime assets cannot reference repair backups or reports")
        paths.add(relative)
    return paths


def scene_asset_hashes(directory: Path) -> dict[str, str]:
    return {
        path.as_posix(): hashlib.sha256((directory / path).read_bytes()).hexdigest()
        for path in sorted(scene_asset_paths(directory))
    }


class SceneBundleSnapshot:
    """Back up scene XML and editable assets, including deleted mesh files."""

    def __init__(self, directory: Path, backup: Path) -> None:
        self.directory = directory
        self.backup = backup
        self.paths = scene_asset_paths(directory) | self._mesh_files() | {Path("scene.xml")}
        if (directory / "metadata.json").is_file():
            self.paths.add(Path("metadata.json"))
        if backup.exists():
            shutil.rmtree(backup)
        self._copy(self.paths, backup)

    def _mesh_files(self) -> set[Path]:
        return {
            path.relative_to(self.directory)
            for path in (self.directory / "meshes").rglob("*") if path.is_file()
        }

    def validate_asset_locations(self) -> None:
        for path in scene_asset_paths(self.directory):
            if path not in self.paths and path.parts[0] != "meshes":
                raise ValueError(f"new repair assets must be written under meshes/: {path}")

    def _copy(self, paths: set[Path], destination: Path) -> None:
        destination.mkdir(parents=True, exist_ok=True)
        for path in sorted(paths):
            source = self.directory / path
            if source.is_file():
                target = destination / path
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)

    def restore(self) -> None:
        current = self.paths | self._mesh_files()
        if (self.directory / "metadata.json").is_file():
            current.add(Path("metadata.json"))
        failed = self.backup.parent / "failed_bundle"
        if failed.exists():
            shutil.rmtree(failed)
        self._copy(current, failed)
        for path in current - self.paths:
            (self.directory / path).unlink(missing_ok=True)
        for path in self.paths:
            target = self.directory / path
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.backup / path, target)
