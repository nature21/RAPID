#!/usr/bin/env python3
"""Build a clickable index for all evaluation galleries under suite_runs/."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
import html
import json
from pathlib import Path
import re

from suites.staged.experiment_paths import experiment_root


ROOT = Path(__file__).resolve().parents[1]
RUN_NAME = re.compile(
    r"^fr3_osc_demo_(?P<task>[1-9][0-9]*)_(?P<trial>[0-9]{3})_eval(?:_attempt)?$"
)
STRATEGY_NAME = re.compile(r"^strategy_(?P<task>[1-9][0-9]*)$")

@dataclass(frozen=True)
class GalleryRecord:
    task: int
    trial: int
    run_name: str
    gallery_path: Path
    relative_link: str
    successes: int
    total: int
    success_rate: float
    variant: str | None
    checkpoint_count: int

    @property
    def is_partial(self) -> bool:
        return self.total != 50


def _load_record(suite_runs: Path, gallery: Path) -> GalleryRecord:
    relative = gallery.relative_to(suite_runs)
    if not relative.parts:
        raise ValueError(f"gallery is not beneath suite_runs: {gallery}")
    run_parts = [index for index, part in enumerate(relative.parts) if RUN_NAME.fullmatch(part)]
    if len(run_parts) != 1:
        raise ValueError(f"cannot resolve eval run from gallery path: {gallery}")
    run_index = run_parts[0]
    run_name = Path(*relative.parts[:run_index + 1]).as_posix()
    run_match = RUN_NAME.fullmatch(relative.parts[run_index])

    strategy_parts = [part for part in relative.parts if STRATEGY_NAME.fullmatch(part)]
    if len(strategy_parts) != 1:
        raise ValueError(f"cannot resolve strategy from gallery path: {gallery}")
    strategy_task = int(STRATEGY_NAME.fullmatch(strategy_parts[0]).group("task"))
    task = int(run_match.group("task"))
    if strategy_task != task:
        raise ValueError(f"run/strategy mismatch in gallery path: {gallery}")

    summary_path = gallery.with_name("summary.json")
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    aggregate = payload.get("summary")
    cases = payload.get("cases")
    if not isinstance(aggregate, dict) or not isinstance(cases, list):
        raise ValueError(f"invalid summary payload: {summary_path}")
    successes = aggregate.get("successes")
    total = aggregate.get("total")
    success_rate = aggregate.get("success_rate")
    if (
        not isinstance(successes, int)
        or not isinstance(total, int)
        or not isinstance(success_rate, (int, float))
        or total < 1
    ):
        raise ValueError(f"invalid success summary: {summary_path}")

    variant = None
    if "_superseded" in relative.parts:
        superseded_index = relative.parts.index("_superseded")
        if superseded_index + 1 < len(relative.parts):
            variant = relative.parts[superseded_index + 1]
        else:
            variant = "superseded"
    checkpoints = {
        str(case["checkpoint_sha256"])
        for case in cases
        if isinstance(case, dict) and case.get("checkpoint_sha256")
    }
    return GalleryRecord(
        task=task,
        trial=int(run_match.group("trial")),
        run_name=run_name,
        gallery_path=gallery,
        relative_link=relative.as_posix(),
        successes=successes,
        total=total,
        success_rate=float(success_rate),
        variant=variant,
        checkpoint_count=len(checkpoints),
    )


def discover_galleries(suite_runs: Path) -> list[GalleryRecord]:
    records = [
        _load_record(suite_runs, path)
        for path in suite_runs.rglob("trajectory_gallery.html")
        if "_superseded" not in path.relative_to(suite_runs).parts
    ]
    return sorted(
        records,
        key=lambda item: (
            item.task,
            item.trial,
            item.variant is not None,
            item.variant or "",
        ),
    )


def _rate_class(rate: float) -> str:
    if rate >= 0.9:
        return "excellent"
    if rate >= 0.75:
        return "good"
    if rate >= 0.6:
        return "fair"
    return "low"


def _format_percent(percent: float) -> str:
    if percent.is_integer():
        return f"{percent:.0f}"
    return f"{percent:.1f}"


def _panel(record: GalleryRecord) -> str:
    badges: list[str] = []
    if record.run_name.endswith("_eval_attempt"):
        badges.append('<span class="badge">attempt evaluation</span>')
    if record.variant is not None:
        badges.append('<span class="badge historical">superseded</span>')
    if record.is_partial:
        badges.append(f'<span class="badge partial">partial · {record.total} cases</span>')
    if record.checkpoint_count > 1:
        badges.append(
            f'<span class="badge warning">{record.checkpoint_count} checkpoints</span>'
        )
    badge_row = f'<div class="badges">{"".join(badges)}</div>' if badges else ""
    variant = (
        f'<p class="variant">{html.escape(record.variant.replace("_", " "))}</p>'
        if record.variant
        else '<p class="variant">current result</p>'
    )
    percent = record.success_rate * 100.0
    percent_label = _format_percent(percent)
    return f"""
<a class="panel {_rate_class(record.success_rate)}" href="{html.escape(record.relative_link)}">
  <div class="panel-top">
    <div><span class="eyebrow">Trial</span><h3>{record.trial:03d}</h3></div>
    <div class="rate">{percent_label}<span>%</span></div>
  </div>
  {variant}
  {badge_row}
  <div class="bar" role="img" aria-label="{percent:.1f}% success">
    <span style="width:{max(0.0, min(percent, 100.0)):.2f}%"></span>
  </div>
  <div class="panel-bottom">
    <strong>{record.successes}/{record.total}</strong><span>successful</span>
    <span class="open">Open gallery&nbsp;→</span>
  </div>
</a>"""


def build_document(records: list[GalleryRecord]) -> str:
    total = sum(record.total for record in records)
    successes = sum(record.successes for record in records)
    overall = f"{successes / total:.1%}" if total else "—"
    snapshot = (
        f"Snapshot: {successes}/{total} recorded cases successful ({overall}). "
        "Partial results are provisional. Regenerate this index to update its counts."
    )
    by_task: dict[int, list[GalleryRecord]] = {}
    for record in records:
        by_task.setdefault(record.task, []).append(record)
    sections: list[str] = []
    for task, task_records in sorted(by_task.items()):
        best = max(task_records, key=lambda item: item.success_rate)
        best_percent = _format_percent(best.success_rate * 100.0)
        panels = "".join(_panel(record) for record in task_records)
        sections.append(
            f"""
<section id="task-{task}">
  <div class="section-heading">
    <div><span class="eyebrow">Benchmark task</span><h2>Task {task}</h2></div>
    <p>{len(task_records)} trials · best {best_percent}%</p>
  </div>
  <div class="grid">{panels}</div>
</section>"""
        )
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>RAPID Suite Runs</title>
<style>
:root{{--bg:#0a0c10;--surface:#12161d;--surface-2:#191f29;--line:#29313d;--text:#f4f7fb;--muted:#98a5b5;--accent:#79a8ff}}
*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(circle at 20% -10%,#1c2f4a 0,transparent 36%),var(--bg);color:var(--text);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}}
main{{width:min(1500px,calc(100% - 40px));margin:0 auto;padding:56px 0 80px}}
.hero{{display:flex;justify-content:space-between;gap:32px;align-items:flex-end;margin-bottom:50px}}
.hero h1{{font-size:clamp(2.3rem,5vw,4.8rem);letter-spacing:-.055em;line-height:.93;margin:9px 0 14px}}
.hero p{{color:var(--muted);margin:0;max-width:680px;font-size:1.03rem}}
.summary{{display:flex;gap:28px;background:#111821cc;border:1px solid var(--line);border-radius:18px;padding:18px 22px;backdrop-filter:blur(12px)}}
.summary strong{{display:block;font-size:1.65rem}}.summary span{{color:var(--muted);font-size:.78rem;text-transform:uppercase;letter-spacing:.12em}}
.eyebrow{{color:var(--accent);font-size:.7rem;font-weight:750;letter-spacing:.16em;text-transform:uppercase}}
section{{margin-top:48px;scroll-margin-top:24px}}
.section-heading{{display:flex;align-items:flex-end;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:13px;margin-bottom:18px}}
.section-heading h2{{font-size:1.8rem;margin:4px 0 0;letter-spacing:-.03em}}.section-heading p{{color:var(--muted);margin:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(245px,1fr));gap:15px}}
.panel{{--tone:#79a8ff;display:block;min-height:236px;padding:20px;border:1px solid var(--line);border-radius:17px;background:linear-gradient(145deg,var(--surface-2),var(--surface));color:inherit;text-decoration:none;transition:transform .16s ease,border-color .16s ease,box-shadow .16s ease}}
.panel:hover{{transform:translateY(-3px);border-color:var(--tone);box-shadow:0 16px 38px #0007}}
.panel.excellent{{--tone:#59d48a}}.panel.good{{--tone:#79a8ff}}.panel.fair{{--tone:#f0b45b}}.panel.low{{--tone:#ee7777}}
.panel-top{{display:flex;align-items:flex-start;justify-content:space-between}}.panel h3{{font-size:2rem;line-height:1;margin:4px 0 0;letter-spacing:-.04em}}
.rate{{color:var(--tone);font-size:2.8rem;font-weight:780;letter-spacing:-.06em;line-height:.9}}.rate span{{font-size:1rem;letter-spacing:0;margin-left:2px}}
.variant{{height:34px;color:var(--muted);font-size:.78rem;margin:14px 0 9px;overflow:hidden}}
.badges{{display:flex;flex-wrap:wrap;gap:6px;margin:-4px 0 9px}}.badge{{font-size:.65rem;font-weight:720;letter-spacing:.04em;border-radius:99px;padding:4px 7px;background:#263142;color:#cbd7e6}}.badge.historical{{background:#382d4b;color:#d8bff8}}.badge.partial{{background:#3b3021;color:#f3cf96}}.badge.warning{{background:#48272b;color:#ffb7bf}}
.bar{{height:7px;border-radius:99px;background:#282f39;overflow:hidden;margin:14px 0}}.bar span{{display:block;height:100%;border-radius:inherit;background:var(--tone)}}
.panel-bottom{{display:grid;grid-template-columns:auto 1fr;align-items:baseline;gap:0 6px}}.panel-bottom strong{{font-size:1.05rem}}.panel-bottom>span:not(.open){{color:var(--muted);font-size:.78rem}}.open{{grid-column:1/-1;color:var(--tone);font-size:.78rem;margin-top:18px}}
footer{{color:var(--muted);font-size:.76rem;margin-top:54px;border-top:1px solid var(--line);padding-top:18px}}
@media(max-width:720px){{main{{width:min(100% - 24px,1500px);padding-top:32px}}.hero{{display:block}}.summary{{margin-top:24px;width:max-content}}.grid{{grid-template-columns:1fr 1fr}}}}
@media(max-width:480px){{.grid{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<header class="hero">
  <div><span class="eyebrow">RAPID evaluation archive</span><h1>Suite Runs</h1><p>Success rates across every task and trial. Select a panel to open its trajectory gallery.</p><p>{snapshot}</p></div>
  <div class="summary"><div><strong>{len(by_task)}</strong><span>tasks</span></div><div><strong>{len(records)}</strong><span>panels</span></div></div>
</header>
{"".join(sections)}
<footer>Generated {generated} · Data read from each gallery's summary.json</footer>
</main></body>
</html>
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-runs",
        "--experiment-root",
        type=Path,
        default=None,
        help="Experiment directory to scan (default: RAPID_EXPERIMENT_ROOT or suite_runs)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: SUITE_RUNS/index.html)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    suite_runs = experiment_root(ROOT, args.suite_runs)
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else suite_runs / "index.html"
    )
    records = discover_galleries(suite_runs)
    if not records:
        raise RuntimeError(f"no trajectory galleries found beneath {suite_runs}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_document(records), encoding="utf-8")
    print(
        f"Wrote {output} with {len(records)} panels across "
        f"{len({record.task for record in records})} tasks."
    )


if __name__ == "__main__":
    main()
