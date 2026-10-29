"""Unified Real2Sim reconstruction, packaging, and viewing CLI."""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from real2sim.cli import add_pipeline_options, run_pipeline_from_args
from vistools.rgbd_input import is_rgbd_scene_directory, load_rgbd_scene


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


@dataclass(frozen=True)
class PackagedScene:
    task_dir: Path
    module: str


@dataclass(frozen=True)
class Real2SimBuildResult:
    reconstruction_dir: Path
    packaged_scene: PackagedScene | None
    frame_image: Path | None = None
    preview_image: Path | None = None
    reused_reconstruction: bool = False


def extract_first_frame(
    video_path: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
) -> Path:
    """Extract frame 0 from ``video_path`` into ``output_path`` with ffmpeg."""

    video = Path(video_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"video not found: {video}")
    if output.exists() and not force:
        return output

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to extract the first video frame")

    output.parent.mkdir(parents=True, exist_ok=True)
    tmp_output = output.with_name(f".{output.stem}.tmp{output.suffix or '.png'}")
    if tmp_output.exists():
        tmp_output.unlink()
    try:
        subprocess.run(
            [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(tmp_output),
            ],
            check=True,
        )
        tmp_output.replace(output)
    finally:
        if tmp_output.exists():
            tmp_output.unlink()
    return output


def package_reconstruction(
    source_dir: str | Path,
    *,
    name: str | None = None,
    tasks_root: str | Path | None = None,
    force: bool = False,
) -> PackagedScene:
    """Package a Real2Sim reconstruction as an importable task scene."""

    from real2sim.package_scene import package_scene

    task_dir = package_scene(
        source_dir,
        name=name,
        tasks_root=tasks_root,
        force=force,
    )
    module = f"tasks.invention_real2sim.{task_dir.name}"

    return PackagedScene(
        task_dir=task_dir,
        module=module,
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _snake_identifier(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip()).strip("_").lower()
    name = re.sub(r"_+", "_", name)
    if not name:
        raise ValueError(f"could not derive a task name from {value!r}")
    if name[0].isdigit():
        name = f"scene_{name}"
    if not name.isidentifier():
        raise ValueError(f"not a valid Python package name: {name!r}")
    return name


def _clear_forced_reconstruction_output(
    source: Path,
    output: Path,
    *,
    force: bool,
) -> None:
    """Remove a generated reconstruction bundle before a forced rebuild."""

    if not force or not output.exists():
        return
    source = source.resolve()
    output = output.resolve()
    if output == source or source.is_relative_to(output):
        raise ValueError(
            "refusing to clear a reconstruction output that contains its input: "
            f"{output}"
        )
    protected = {
        Path("/").resolve(),
        Path.home().resolve(),
        _repo_root().resolve(),
        _repo_root().resolve().parent,
    }
    if output in protected:
        raise ValueError(f"refusing to clear broad protected directory: {output}")
    if not output.is_dir():
        raise NotADirectoryError(
            f"reconstruction output exists but is not a directory: {output}"
        )
    shutil.rmtree(output)


def build_from_input(
    input_path: str | Path,
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
) -> Real2SimBuildResult:
    """Run the workflow for an image, video, RGB-D capture, or scene bundle."""

    source = Path(input_path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"input not found: {source}")

    case_name = _snake_identifier(args.name or source.stem)
    reconstruction_dir = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else (_repo_root() / "results" / "real2sim" / case_name).resolve()
    )

    frame_image: Path | None = None
    reused_reconstruction = False
    if source.is_dir():
        if (source / "scene.xml").is_file():
            if args.output is not None:
                parser.error(
                    "--output is only used when reconstructing an image, video, "
                    "or RGB-D capture"
                )
            reconstruction_dir = source
        elif (
            getattr(args, "reuse_existing", False)
            and (reconstruction_dir / "scene.xml").is_file()
        ):
            reused_reconstruction = True
        elif is_rgbd_scene_directory(source):
            rgbd_scene = load_rgbd_scene(source)
            if args.world_frame is None:
                args.world_frame = "robot"
            _clear_forced_reconstruction_output(
                source,
                reconstruction_dir,
                force=args.force,
            )
            run_pipeline_from_args(
                rgbd_scene.image_path,
                reconstruction_dir,
                args,
                parser,
                depth=rgbd_scene.depth,
                depth_metadata=rgbd_scene.camera_metadata,
            )
        else:
            raise FileNotFoundError(
                f"input directory is neither a Real2Sim output bundle nor an RGB-D "
                f"capture; expected scene.xml or color.png, depth.npy, "
                f"intrinsics.json, and one *_eob_calib.yaml file in {source}"
            )
    elif (
        getattr(args, "reuse_existing", False)
        and (reconstruction_dir / "scene.xml").is_file()
    ):
        if source.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
            raise ValueError(
                f"unsupported input type {source.suffix!r}; expected video, image, "
                "RGB-D capture, or Real2Sim output directory"
            )
        if source.suffix.lower() in VIDEO_EXTS:
            existing_frame = (
                Path(args.frame_output).expanduser().resolve()
                if args.frame_output is not None
                else reconstruction_dir / "input_first_frame.png"
            )
            if existing_frame.is_file():
                frame_image = existing_frame
        reused_reconstruction = True
    elif source.suffix.lower() in VIDEO_EXTS:
        _clear_forced_reconstruction_output(
            source,
            reconstruction_dir,
            force=args.force,
        )
        frame_image = (
            Path(args.frame_output).expanduser().resolve()
            if args.frame_output is not None
            else reconstruction_dir / "input_first_frame.png"
        )
        frame_image = extract_first_frame(source, frame_image, force=args.force)
        run_pipeline_from_args(frame_image, reconstruction_dir, args, parser)
    elif source.suffix.lower() in IMAGE_EXTS:
        _clear_forced_reconstruction_output(
            source,
            reconstruction_dir,
            force=args.force,
        )
        run_pipeline_from_args(source, reconstruction_dir, args, parser)
    else:
        raise ValueError(
            f"unsupported input type {source.suffix!r}; expected video, image, "
            "RGB-D capture, or Real2Sim output directory"
        )

    packaged: PackagedScene | None = None
    if not args.no_package:
        packaged = package_reconstruction(
            reconstruction_dir,
            name=case_name,
            tasks_root=args.tasks_root,
            force=args.force,
        )

    preview_image: Path | None = None
    if args.preview or args.viewer:
        if packaged is None:
            parser.error("--preview/--viewer require packaging; remove --no-package")
        assert packaged is not None
        preview_image = _preview_packaged_scene(packaged.module, args)

    return Real2SimBuildResult(
        reconstruction_dir=reconstruction_dir,
        packaged_scene=packaged,
        frame_image=frame_image,
        preview_image=preview_image,
        reused_reconstruction=reused_reconstruction,
    )


def _preview_packaged_scene(module: str, args: argparse.Namespace) -> Path | None:
    from vistools.visualize_scene import launch_scene_viewer, render_scene

    if args.viewer:
        launch_scene_viewer(module)
        return None
    output = (
        Path(args.preview_output).expanduser().resolve()
        if args.preview_output is not None
        else None
    )
    return render_scene(module, output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rapid-real2sim",
        description=(
            "Build a simulation-ready Real2Sim task from a video, image, RGB-D "
            "capture directory, or existing Real2Sim scene bundle."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              ./scripts/rapid-real2sim demo.mp4 --name demo_scene --force
              ./scripts/rapid-real2sim demo.mp4 --name demo_scene --reuse-existing --force
              ./scripts/rapid-real2sim frame.png --name push_scene
              ./scripts/rapid-real2sim real_scenes/scene_1 --name captured_scene
              ./scripts/rapid-real2sim results/real2sim/demo_scene --force
              ./scripts/rapid-real2sim view tasks.invention_real2sim.demo_scene
            """
        ),
    )
    parser.add_argument(
        "input",
        nargs="?",
        help=(
            "Video, RGB image, captured RGB-D directory, or existing Real2Sim "
            "output directory containing scene.xml."
        ),
    )
    workflow_group = parser.add_argument_group("workflow options")
    workflow_group.add_argument(
        "--name",
        default=None,
        help="Case/package name. Defaults to a snake_case version of the input stem.",
    )
    workflow_group.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Reconstruction output directory. Defaults to results/real2sim/<name>.",
    )
    workflow_group.add_argument(
        "--frame-output",
        type=Path,
        default=None,
        help="PNG path for the first frame extracted from a video input.",
    )
    workflow_group.add_argument(
        "--tasks-root",
        type=Path,
        default=None,
        help="Override the packaged task root. Defaults to tasks/invention_real2sim.",
    )
    workflow_group.add_argument(
        "--force",
        action="store_true",
        help=(
            "Clear and rebuild the reconstruction output, then fully replace "
            "existing packaged task directories."
        ),
    )
    workflow_group.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Reuse an existing reconstruction containing scene.xml at the resolved "
            "output path instead of rerunning the image pipeline."
        ),
    )
    workflow_group.add_argument(
        "--no-package",
        action="store_true",
        help="Only reconstruct the Real2Sim scene bundle; do not create a task env.",
    )
    preview_group = workflow_group.add_mutually_exclusive_group()
    preview_group.add_argument(
        "--preview",
        action="store_true",
        help="Render a PNG preview of the packaged task scene.",
    )
    workflow_group.add_argument(
        "--preview-output",
        type=Path,
        default=None,
        help="PNG output path for --preview.",
    )
    preview_group.add_argument(
        "--viewer",
        action="store_true",
        help="Open the packaged task scene in MuJoCo's interactive viewer.",
    )
    reconstruction_group = parser.add_argument_group("reconstruction options")
    add_pipeline_options(reconstruction_group)
    return parser


def _print_build_result(result: Real2SimBuildResult) -> None:
    repo_root = _repo_root()
    reused = " (reused)" if result.reused_reconstruction else ""
    print(
        "[real2sim] reconstruction: "
        f"{_display_path(result.reconstruction_dir, repo_root)}{reused}"
    )
    if result.frame_image is not None:
        print(f"[real2sim] first frame   : {_display_path(result.frame_image, repo_root)}")
    if result.packaged_scene is not None:
        task = result.packaged_scene
        print(f"[real2sim] task package  : {_display_path(task.task_dir, repo_root)}")
        print(f"[real2sim] env module    : {task.module}")
        print(f"[real2sim] agent flag    : --env-module {task.module}")
    if result.preview_image is not None:
        print(f"[real2sim] preview      : {_display_path(result.preview_image, repo_root)}")


def _display_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _main_view(argv: list[str]) -> int:
    from vistools.visualize_scene import main as visualize_main

    return visualize_main(argv, prog="rapid-real2sim view")


def main(argv: list[str] | None = None) -> int:
    args_list = list(sys.argv[1:] if argv is None else argv)
    if args_list and args_list[0] == "view":
        return _main_view(args_list[1:])

    parser = _build_parser()
    args = parser.parse_args(args_list)
    if args.input is None:
        parser.error(
            "pass a video, image, RGB-D capture, or Real2Sim output directory"
        )

    try:
        result = build_from_input(args.input, args, parser)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    _print_build_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
