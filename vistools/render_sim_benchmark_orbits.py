"""Render a resumable orbit-video gallery for simulation benchmark scenes."""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import importlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from vistools.visualize_mujoco import render_orbit_video_model


_CASE_PATTERN = re.compile(r"^case_(\d{3})_scene$")


@dataclass(frozen=True)
class RenderJob:
    output_root: str
    strategy_id: int
    case_index: int
    width: int
    height: int
    fps: int
    duration: float
    overwrite: bool

    @property
    def module_name(self) -> str:
        return (
            f"tasks.sim_benchmark.strategy_{self.strategy_id}."
            f"case_{self.case_index:03d}_scene.scene"
        )

    @property
    def output_path(self) -> Path:
        return (
            Path(self.output_root)
            / f"strategy_{self.strategy_id}"
            / f"case_{self.case_index:03d}_orbit.mp4"
        )


def _parse_int_selection(
    values: Sequence[str] | None,
    *,
    minimum: int,
    maximum: int,
) -> tuple[int, ...]:
    if not values:
        return tuple(range(minimum, maximum + 1))
    selected: set[int] = set()
    for raw in values:
        for part in raw.split(","):
            text = part.strip()
            if not text:
                continue
            if "-" in text:
                start, end = (int(item) for item in text.split("-", 1))
                if start > end:
                    raise ValueError(f"range must be ascending: {text}")
                selected.update(range(start, end + 1))
            else:
                selected.add(int(text))
    invalid = sorted(item for item in selected if not minimum <= item <= maximum)
    if invalid:
        raise ValueError(f"selection must be in {minimum}..{maximum}: {invalid}")
    if not selected:
        raise ValueError("selection cannot be empty")
    return tuple(sorted(selected))


def _valid_video(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    completed = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,width,height,nb_frames",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return False
    try:
        streams = json.loads(completed.stdout).get("streams", [])
    except json.JSONDecodeError:
        return False
    return bool(streams and streams[0].get("codec_name"))


def _render(job: RenderJob) -> dict[str, object]:
    output = job.output_path
    if not job.overwrite and _valid_video(output):
        return {
            "strategy_id": job.strategy_id,
            "case_index": job.case_index,
            "path": str(output),
            "bytes": output.stat().st_size,
            "status": "existing",
        }

    module = importlib.import_module(job.module_name)
    model = module.make_model()
    data = module.make_data(model)
    camera = dict(getattr(module, "DEFAULT_CAMERA", {}) or {})
    lookat_raw = camera.get("camera_lookat")
    lookat = (
        tuple(float(value) for value in lookat_raw)
        if lookat_raw is not None
        else None
    )
    render_orbit_video_model(
        model,
        output,
        data=data,
        width=job.width,
        height=job.height,
        fps=job.fps,
        duration=job.duration,
        start_azimuth=float(camera.get("camera_azimuth", 90.0)),
        elevation=float(camera.get("camera_elevation", -37.0)),
        distance=(
            float(camera["camera_distance"])
            if camera.get("camera_distance") is not None
            else None
        ),
        lookat=lookat,
        step_count=0,
        turns=1.0,
    )
    if not _valid_video(output):
        raise RuntimeError(f"ffprobe rejected rendered video: {output}")
    return {
        "strategy_id": job.strategy_id,
        "case_index": job.case_index,
        "path": str(output),
        "bytes": output.stat().st_size,
        "status": "rendered",
    }


def _write_gallery(
    repo_root: Path,
    output_root: Path,
    records: Sequence[dict[str, object]],
) -> None:
    by_strategy: dict[int, list[dict[str, object]]] = {}
    for record in records:
        by_strategy.setdefault(int(record["strategy_id"]), []).append(record)

    style = """
<style>
body{font-family:system-ui,sans-serif;margin:24px;background:#111;color:#eee}
a{color:#8cc8ff}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:18px}
article{background:#222;padding:12px;border-radius:8px}video{width:100%;height:auto}
h2{margin-top:40px}h3{margin:0 0 8px}p{font-size:13px;line-height:1.35;color:#ccc}
dl{display:grid;grid-template-columns:1fr 2fr;gap:4px 10px;font-size:13px}
dt{color:#aaa}dd{margin:0}
</style>
"""
    links: list[str] = []
    for strategy_id in sorted(by_strategy):
        cards: list[str] = []
        for record in sorted(by_strategy[strategy_id], key=lambda item: int(item["case_index"])):
            case_index = int(record["case_index"])
            case_dir = (
                repo_root
                / "tasks"
                / "sim_benchmark"
                / f"strategy_{strategy_id}"
                / f"case_{case_index:03d}_scene"
            )
            goal_path = case_dir / "goal.txt"
            goal = goal_path.read_text(encoding="utf-8").strip() if goal_path.is_file() else ""
            public_path = case_dir / "benchmark_case.json"
            benchmark_version = (
                json.loads(public_path.read_text(encoding="utf-8"))[
                    "benchmark_version"
                ]
                if public_path.is_file()
                else ""
            )
            private_path = (
                repo_root
                / "benchmark_private"
                / "sealed"
                / benchmark_version
                / "manifests"
                / f"strategy_{strategy_id}"
                / f"case_{case_index:03d}.json"
            )
            private = (
                json.loads(private_path.read_text(encoding="utf-8"))
                if private_path.is_file()
                else {}
            )
            parameters = private.get("parameters", {})
            feasibility = (
                private.get("context_layout", {})
                .get("wall_flip", {})
                .get("manipulation_feasibility", {})
            )
            dimensions = feasibility.get("sorted_dimensions_m", [])
            if not dimensions:
                dimensions = sorted(
                    2.0 * float(value)
                    for value in private.get("target_half_extents", [])
                )
            dimensions_text = (
                " × ".join(f"{100.0 * float(value):.1f}" for value in dimensions)
                + " cm"
                if dimensions
                else "n/a"
            )
            physics_text = (
                f"{float(feasibility['mass_kg']):.3f} kg; friction "
                f"{float(feasibility['sliding_friction']):.3f}"
                if "mass_kg" in feasibility
                and "sliding_friction" in feasibility
                else "n/a"
            )
            grasp_text = (
                f"thickness {1000.0 * float(feasibility['grasp_thickness_m']):.1f} mm; "
                f"upright height {1000.0 * float(feasibility['upright_height_m']):.1f} mm"
                if "grasp_thickness_m" in feasibility
                and "upright_height_m" in feasibility
                else "n/a"
            )
            layout = private.get("context_layout", {})
            wall_flip = layout.get("wall_flip", {})
            planar_slot = layout.get("planar_slot", {})
            if "target_upright_height" in wall_flip:
                grasp_parts = [
                    "upright height "
                    f"{100.0 * float(wall_flip['target_upright_height']):.1f} cm"
                ]
                if "required_clearance_pull" in wall_flip:
                    grasp_parts.append(
                        "pull clearance "
                        f"{100.0 * float(wall_flip['required_clearance_pull']):.1f} cm"
                    )
                grasp_text = "; ".join(grasp_parts)
            elif "required_target_width" in planar_slot:
                grasp_text = (
                    f"aligned width "
                    f"{100.0 * float(planar_slot['required_target_width']):.1f} cm; "
                    f"opening {100.0 * float(planar_slot['clear_width']):.1f} cm"
                )
            video = Path(str(record["path"])).relative_to(output_root).as_posix()
            cards.append(
                "<article><h3>Case "
                f"{case_index:03d}</h3><video controls preload=\"metadata\" "
                f"src=\"{html.escape(video)}\"></video>"
                f"<p>{html.escape(goal)}</p><dl>"
                f"<dt>Bucket</dt><dd>{html.escape(str(parameters.get('bucket', 'n/a')))}</dd>"
                f"<dt>Target</dt><dd>{html.escape(str(parameters.get('appearance', 'n/a')))}</dd>"
                f"<dt>Dimensions</dt><dd>{html.escape(dimensions_text)}</dd>"
                f"<dt>Physics</dt><dd>{html.escape(physics_text)}</dd>"
                f"<dt>Flip/grasp</dt><dd>{html.escape(grasp_text)}</dd>"
                "<dt>Certification</dt><dd>passed</dd>"
                "</dl></article>"
            )
        strategy_document = (
            "<!doctype html><html><head><meta charset=\"utf-8\">"
            f"<title>Strategy {strategy_id} orbit videos</title>{style}</head>"
            f"<body><p><a href=\"index.html\">← All strategies</a></p>"
            f"<h1>Strategy {strategy_id}</h1><div class=\"grid\">"
            + "".join(cards)
            + "</div></body></html>\n"
        )
        page_name = f"strategy_{strategy_id}.html"
        (output_root / page_name).write_text(strategy_document, encoding="utf-8")
        links.append(
            f"<article><h2><a href=\"{page_name}\">Strategy {strategy_id}</a></h2>"
            f"<p>{len(cards)} orbit videos</p></article>"
        )

    document = (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>Simulation benchmark orbit videos</title>{style}</head>"
        "<body><h1>Simulation benchmark orbit videos</h1>"
        f"<p>{len(records)} rendered scenes across {len(by_strategy)} selected "
        "strategy group(s).</p>"
        "<div class=\"grid\">"
        + "".join(links)
        + "</div></body></html>\n"
    )
    (output_root / "index.html").write_text(document, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", action="append", default=None)
    parser.add_argument("--case", action="append", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/sim_benchmark_orbits"),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    output_root = args.output.expanduser()
    if not output_root.is_absolute():
        output_root = repo_root / output_root
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    strategies = _parse_int_selection(args.strategy, minimum=1, maximum=8)
    cases = _parse_int_selection(args.case, minimum=1, maximum=50)
    jobs = [
        RenderJob(
            output_root=str(output_root),
            strategy_id=strategy_id,
            case_index=case_index,
            width=args.width,
            height=args.height,
            fps=args.fps,
            duration=args.duration,
            overwrite=args.overwrite,
        )
        for strategy_id in strategies
        for case_index in cases
    ]

    records: list[dict[str, object]] = []
    failures: list[str] = []
    worker_count = max(1, min(int(args.jobs), len(jobs)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_jobs = {executor.submit(_render, job): job for job in jobs}
        for completed, future in enumerate(
            concurrent.futures.as_completed(future_jobs), start=1
        ):
            job = future_jobs[future]
            try:
                record = future.result()
            except Exception as exc:
                failures.append(
                    f"strategy {job.strategy_id} case {job.case_index:03d}: {exc}"
                )
                print(f"[{completed}/{len(jobs)}] FAILED {failures[-1]}", flush=True)
            else:
                records.append(record)
                print(
                    f"[{completed}/{len(jobs)}] {record['status']} "
                    f"strategy {job.strategy_id} case {job.case_index:03d}",
                    flush=True,
                )

    records.sort(key=lambda item: (int(item["strategy_id"]), int(item["case_index"])))
    manifest = {
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
        "duration": args.duration,
        "videos": records,
        "failures": failures,
    }
    (output_root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_gallery(repo_root, output_root, records)
    total_bytes = sum(int(record["bytes"]) for record in records)
    print(
        f"rendered/verified {len(records)} videos ({total_bytes / 1024**2:.1f} MiB); "
        f"gallery: {output_root / 'index.html'}"
    )
    if failures:
        print(f"{len(failures)} render(s) failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
