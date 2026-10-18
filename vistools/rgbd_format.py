"""File names and depth previews for calibrated RGB-D demonstrations."""

import numpy as np


CALIBRATION_FILENAME = "panda_eob_calib.yaml"
RGB_VIDEO_FILENAME = "rgb.mp4"
DEPTH_VIDEO_FILENAME = "depth.mkv"
DEPTH_PREVIEW_FILENAME = "depth_preview.mp4"
TIMESTAMPS_FILENAME = "timestamps.csv"
CAPTURE_METADATA_FILENAME = "capture_metadata.json"


def _depth_visualization(depth: np.ndarray) -> np.ndarray:
    """Create a robust 8-bit preview without altering the saved metric depth."""

    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    preview = np.zeros(depth.shape, dtype=np.uint8)
    if not np.any(valid):
        return preview
    low, high = np.percentile(depth[valid], [1.0, 99.0])
    if high <= low:
        preview[valid] = 255
        return preview
    normalized = np.clip((depth - low) / (high - low), 0.0, 1.0)
    preview[valid] = np.asarray(normalized[valid] * 255.0, dtype=np.uint8)
    return preview

