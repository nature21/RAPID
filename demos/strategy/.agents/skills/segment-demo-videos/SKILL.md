---
name: segment-demo-videos
description: Annotate and split calibrated, timestamp-synchronized RGB-D demonstrations in demos/strategy. Use when creating primitive-learning phase clips, updating segments.csv after visually inspecting a demo, or regenerating lean segment_N directories with first-frame metric depth and camera calibration.
---

# Segment Demo Videos

Create primitive-level clips without changing the source capture. Keep complete,
contiguous annotations in `segments.csv`, then use the provided generator to
publish reconstructable segment directories only after every output validates.

## Annotate the demonstrations

1. Inspect each `demo_N/rgb.mp4` visually. Use contact changes, release and
   recontact, fixture use, and changes in motion objective as phase boundaries.
2. Write one row per phase to `demos/strategy/segments.csv` using exactly:
   `demo,phase,start_s,end_s,primitive,phase_summary,contact_mode,constraints_strategy`.
3. Start phase numbering at 1. Set the first `start_s` to `0`, make every
   `end_s` equal the next `start_s`, and set the final `end_s` to the complete
   RGB video duration. Do not leave gaps or overlaps.
4. Describe the physical contact and environmental constraint, not only the
   visible hand trajectory. Use stable snake_case primitive names.

## Generate segments

From the repository root, process all demos with:

```bash
./scripts/rapid-python demos/strategy/.agents/skills/segment-demo-videos/scripts/segment_demo.py \
  --root demos/strategy \
  --manifest demos/strategy/segments.csv \
  --force
```

For one or several demos, repeat `--demo-dir` instead of using `--root`. Omit
`--force` when outputs must not already exist.

The generator maps phase times through the recorded timestamps, clips only
`rgb.mp4`, extracts the selected first frame from the source lossless
`depth.mkv`, copies intrinsics and eye-on-base calibration, loads the result
through the production RGB-D loader, and checks RGB/timestamp counts. It stages
the complete output set before replacing any existing segments.

## Output contract

Each `demo_N/segment_J` contains only:

- `rgb.mp4`
- `timestamps.csv`, rebased to zero with source frame/time provenance
- `color.png`, decoded from the first segment RGB frame
- `depth.npy`, metric float32 depth for the first source depth frame
- `depth.png`, a view-only rendering of that metric depth
- `intrinsics.json`
- `panda_eob_calib.yaml`
- `capture_metadata.json`, including annotation, calibration, and source ranges

Never add `depth.mkv` or `depth_preview.mp4` to a segment. Keep both in the
source demo because later segment regeneration depends on lossless depth.

## Validate annotations and outputs

Treat generator failures as annotation or source-integrity errors; do not weaken
coverage or synchronization checks.
