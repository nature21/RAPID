import argparse
import base64
import io
import os

import numpy as np
import requests
from PIL import Image

from real2sim.vfm_clients.config import get_endpoint


DEFAULT_HOST, DEFAULT_PORT = get_endpoint("depth_anything")
DEFAULT_SERVER_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/da3_query"


def encode_image(image):
    """Return a base64 PNG encoding for a PIL image, numpy image, or image path."""
    image = _as_rgb_image(image)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def decode_array(field):
    """Reconstruct a numpy array from the server's {data, shape, dtype} dict."""
    raw = base64.b64decode(field["data"])
    arr = np.frombuffer(raw, dtype=np.dtype(field["dtype"]))
    return arr.reshape(field["shape"])


def depth_anything_request(
    image,
    focal=None,
    intrinsics=None,
    extrinsics=None,
    use_ray_pose=None,
    process_res=None,
    process_res_method=None,
    server_url=DEFAULT_SERVER_URL,
    timeout=600,
    return_metadata=False,
):
    """
    Query the Depth Anything 3 server and return the decoded depth map.

    Args:
        image: PIL.Image, HxWxC numpy array, or image file path.
        focal: Optional pixel focal length for metric conversion.
        intrinsics: Optional 3x3 camera intrinsic matrix.
        extrinsics: Optional 3x4 or 4x4 OpenCV/Colmap w2c camera matrix.
        use_ray_pose: Optional bool to request DA3 ray-based pose estimation.
        process_res: Optional DA3 processing resolution.
        process_res_method: Optional DA3 input resize/crop method.
        server_url: Full URL for the DA3 endpoint.
        timeout: Request timeout in seconds.
        return_metadata: If True, return (depth, metadata).

    Returns:
        np.ndarray: A float32 depth map with shape (H, W), or
        (np.ndarray, dict) when return_metadata=True.
    """
    image = _as_rgb_image(image)
    payload = {"rgb_image": encode_image(image)}

    if focal is not None:
        payload["focal"] = float(focal)
    if intrinsics is not None:
        payload["intrinsics"] = _normalize_intrinsics(intrinsics)
    if extrinsics is not None:
        payload["extrinsics"] = _normalize_extrinsics(extrinsics)
    if use_ray_pose is not None:
        payload["use_ray_pose"] = bool(use_ray_pose)
    if process_res is not None:
        payload["process_res"] = int(process_res)
    if process_res_method is not None:
        payload["process_res_method"] = process_res_method

    try:
        response = requests.post(server_url, json=payload, timeout=timeout)
    except requests.exceptions.RequestException as exc:
        raise ConnectionError(f"Could not connect to DA3 server at {server_url}: {exc}") from exc
    try:
        response_json = response.json()
    except ValueError as exc:
        raise ConnectionError(
            f"Depth Anything server returned non-JSON response: {response.text[:500]}"
        ) from exc

    if response.status_code != 200:
        error = response_json.get("error", response.text)
        raise ConnectionError(f"Depth Anything request failed: {error}")

    if "depth" not in response_json:
        raise ValueError("Depth Anything response did not include a depth field")

    depth = decode_array(response_json["depth"]).astype(np.float32, copy=False)
    if depth.ndim != 2:
        raise ValueError(f"depth should be (H, W), got {depth.shape}")

    width, height = image.size
    if depth.shape != (height, width):
        raise ValueError(
            f"depth shape {depth.shape} should match image (H, W)={(height, width)}"
        )

    if not return_metadata:
        return depth

    metadata = {
        key: value
        for key, value in response_json.items()
        if key not in {"depth", "conf", "intrinsics", "extrinsics"}
    }
    for key in ("conf", "intrinsics", "extrinsics"):
        if key in response_json:
            metadata[key] = decode_array(response_json[key])

    return depth, metadata


def save_depth_visualization(depth, output_path):
    vis = ((depth - depth.min()) / (depth.max() - depth.min() + 1e-8) * 255).astype(
        np.uint8
    )

    try:
        import cv2

        vis = cv2.applyColorMap(vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(output_path, vis)
    except ImportError:
        Image.fromarray(vis).save(output_path)


def _as_rgb_image(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")

    if isinstance(image, (str, os.PathLike)):
        return Image.open(image).convert("RGB")

    raise TypeError(
        "image must be a PIL.Image, numpy.ndarray, or filesystem path; "
        f"got {type(image).__name__}"
    )


def _normalize_intrinsics(intrinsics):
    intrinsics = np.asarray(intrinsics, dtype=float)
    if intrinsics.shape != (3, 3):
        raise ValueError(f"intrinsics must have shape (3, 3), got {intrinsics.shape}")
    return intrinsics.tolist()


def _normalize_extrinsics(extrinsics):
    extrinsics = np.asarray(extrinsics, dtype=float)
    if extrinsics.shape not in ((3, 4), (4, 4)):
        raise ValueError(
            f"extrinsics must have shape (3, 4) or (4, 4), got {extrinsics.shape}"
        )
    return extrinsics.tolist()


def _parse_intrinsics(value):
    vals = [float(v) for v in value.split(",")]
    if len(vals) != 9:
        raise ValueError("intrinsics must have 9 comma-separated values")
    return [vals[0:3], vals[3:6], vals[6:9]]


def _parse_extrinsics(value):
    vals = [float(v) for v in value.split(",")]
    if len(vals) == 12:
        return [vals[0:4], vals[4:8], vals[8:12]]
    if len(vals) == 16:
        return [vals[0:4], vals[4:8], vals[8:12], vals[12:16]]
    raise ValueError("extrinsics must have 12 or 16 comma-separated values")


def save_result_npz(depth, metadata, output_path):
    arrays = {"depth": depth}
    for name in ("conf", "intrinsics", "extrinsics"):
        if name in metadata:
            arrays[name] = metadata[name]
    np.savez_compressed(output_path, **arrays)


def print_camera_parameters(metadata):
    if "intrinsics" in metadata:
        print("estimated intrinsics K:")
        print(np.array2string(metadata["intrinsics"], precision=4, suppress_small=True))
        print("intrinsics source:", metadata.get("intrinsics_source", "unknown"))
    else:
        print("intrinsics: not returned by server")

    if "extrinsics" in metadata:
        convention = metadata.get("extrinsics_convention", "opencv_w2c_3x4")
        print(f"estimated extrinsics ({convention}):")
        print(np.array2string(metadata["extrinsics"], precision=4, suppress_small=True))
        print("extrinsics source:", metadata.get("extrinsics_source", "unknown"))
    else:
        print("extrinsics: not returned by server")


def main():
    parser = argparse.ArgumentParser(
        description="Query the DA3 API for single-image depth and camera parameters"
    )
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", default=DEFAULT_PORT, type=int)
    parser.add_argument("--url", default=None, help="Full DA3 endpoint URL.")
    parser.add_argument(
        "--image",
        default=os.path.join(os.path.dirname(__file__), "example.jpg"),
    )
    parser.add_argument(
        "--output",
        default=os.path.join(os.path.dirname(__file__), "da3_depth_vis_api.png"),
    )
    parser.add_argument(
        "--npz-output",
        default=os.path.join(os.path.dirname(__file__), "da3_single_image_result.npz"),
        help="Compressed npz file for depth/conf/intrinsics/extrinsics.",
    )
    parser.add_argument(
        "--focal",
        default=None,
        type=float,
        help="Pixel focal length for metric conversion.",
    )
    parser.add_argument(
        "--intrinsics",
        default=None,
        type=str,
        help="Comma-separated 9 values of the 3x3 K matrix in row-major order.",
    )
    parser.add_argument(
        "--extrinsics",
        default=None,
        type=str,
        help="Comma-separated 12 or 16 values of a 3x4/4x4 w2c matrix in row-major order.",
    )
    parser.add_argument(
        "--use-ray-pose",
        action="store_true",
        default=None,
        help="Ask DA3 to use ray-based pose estimation.",
    )
    parser.add_argument(
        "--process-res",
        default=None,
        type=int,
        help="Optional DA3 processing resolution override.",
    )
    parser.add_argument(
        "--process-res-method",
        default=None,
        choices=[
            "upper_bound_resize",
            "upper_bound_crop",
            "lower_bound_resize",
            "lower_bound_crop",
        ],
        help="Optional DA3 input resize/crop method override.",
    )
    parser.add_argument("--timeout", default=600, type=float)
    args = parser.parse_args()

    server_url = args.url or f"http://{args.host}:{args.port}/da3_query"
    if not os.path.isfile(args.image):
        raise FileNotFoundError(f"image not found: {args.image}")

    image = _as_rgb_image(args.image)
    print(f"Sending {args.image} ({image.size[0]}x{image.size[1]}) to {server_url}")

    intrinsics = _parse_intrinsics(args.intrinsics) if args.intrinsics else None
    extrinsics = _parse_extrinsics(args.extrinsics) if args.extrinsics else None
    depth, metadata = depth_anything_request(
        image,
        focal=args.focal,
        intrinsics=intrinsics,
        extrinsics=extrinsics,
        use_ray_pose=args.use_ray_pose,
        process_res=args.process_res,
        process_res_method=args.process_res_method,
        server_url=server_url,
        timeout=args.timeout,
        return_metadata=True,
    )

    units = metadata.get("units", "canonical")
    print(
        "is_metric:",
        metadata.get("is_metric"),
        "| units:",
        units,
        "| focal_px:",
        metadata.get("focal_px"),
    )
    print("model:", metadata.get("model_id") or metadata.get("model"))
    print("pose mode use_ray_pose:", metadata.get("use_ray_pose"))
    print("depth shape:", depth.shape, "dtype:", depth.dtype)
    print(
        f"depth stats ({units}): min={depth.min():.3f} "
        f"max={depth.max():.3f} mean={depth.mean():.3f}"
    )

    if args.focal is not None or intrinsics is not None:
        if units != "meters":
            raise ValueError(f"expected metric depth but got units={units}")
        print("metric conversion applied")

    server_stats = metadata.get("depth_stats", {})
    if server_stats:
        if abs(server_stats["min"] - float(depth.min())) >= 1e-2:
            raise ValueError("server min stat does not match decoded depth")
        if abs(server_stats["max"] - float(depth.max())) >= 1e-2:
            raise ValueError("server max stat does not match decoded depth")
        print("server stats match decoded depth")

    for name in ("conf", "intrinsics", "extrinsics"):
        if name in metadata:
            arr = metadata[name]
            print(f"{name}: shape={arr.shape} dtype={arr.dtype}")
    print_camera_parameters(metadata)

    save_depth_visualization(depth, args.output)
    print("Saved depth visualization to", args.output)
    save_result_npz(depth, metadata, args.npz_output)
    print("Saved decoded depth/camera arrays to", args.npz_output)
    print("DA3 API test passed")


if __name__ == "__main__":
    main()
