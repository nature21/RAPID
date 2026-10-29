"""Depth Anything 3 depth and camera service for uncalibrated RGB inputs."""

import numpy as np
from PIL import Image

from server.common import create_app, decode_image, encode_array, parser, serve


DEFAULT_MODEL = "depth-anything/DA3NESTED-GIANT-LARGE-1.1"


def resize_map(value, size):
    return np.asarray(Image.fromarray(np.asarray(value, dtype=np.float32)).resize(
        size, Image.Resampling.BILINEAR), dtype=np.float32)


def prediction_response(prediction, image, model_name, intrinsics=None, extrinsics=None, focal=None):
    depth = np.asarray(prediction.depth[0], dtype=np.float32)
    height, width = depth.shape
    depth = resize_map(depth, image.size)
    metric = bool(np.asarray(getattr(prediction, "is_metric", False)).any())
    focal_px = float(focal) if focal is not None else (
        float((intrinsics[0, 0] + intrinsics[1, 1]) / 2) if intrinsics is not None else None)
    canonical = "da3metric" in model_name.lower()
    if not metric and canonical and focal_px is not None:
        depth = depth * (focal_px / 300.0)
        metric = True
    result = {
        "depth": encode_array(depth), "is_metric": int(metric),
        "units": "meters" if metric else ("canonical" if canonical else "relative"),
        "model": model_name, "focal_px": focal_px,
        "extrinsics_convention": "opencv_w2c_3x4",
    }
    if getattr(prediction, "conf", None) is not None:
        result["conf"] = encode_array(resize_map(prediction.conf[0], image.size))
    estimated_k = getattr(prediction, "intrinsics", None)
    if estimated_k is not None:
        intrinsics = np.asarray(estimated_k[0], dtype=np.float32).copy()
        intrinsics[0] *= image.width / width
        intrinsics[1] *= image.height / height
    if intrinsics is not None:
        result["intrinsics"] = encode_array(np.asarray(intrinsics, dtype=np.float32))
        result["intrinsics_source"] = "estimated" if estimated_k is not None else "request"
    estimated_e = getattr(prediction, "extrinsics", None)
    if estimated_e is not None:
        extrinsics = estimated_e[0]
    if extrinsics is not None:
        extrinsics = np.asarray(extrinsics, dtype=np.float32)[:3, :4]
        result["extrinsics"] = encode_array(extrinsics)
        result["extrinsics_source"] = "estimated" if estimated_e is not None else "request"
    return result


class DepthAnything:
    def __init__(self, model_id, device):
        from depth_anything_3.api import DepthAnything3

        self.model_id = model_id
        self.model = DepthAnything3.from_pretrained(model_id).to(device)

    def __call__(self, image, **options):
        import torch

        intrinsics, extrinsics = options["intrinsics"], options["extrinsics"]
        with torch.inference_mode():
            prediction = self.model.inference(
                [np.asarray(image)],
                intrinsics=None if intrinsics is None else intrinsics[None],
                extrinsics=None if extrinsics is None else extrinsics[None],
                use_ray_pose=options["use_ray_pose"], process_res=options["process_res"],
                process_res_method=options["process_res_method"],
            )
        result = prediction_response(prediction, image, self.model.model_name,
                                     intrinsics, extrinsics, options["focal"])
        result["model_id"] = self.model_id
        return result


def build_app(predictor):
    def query(payload):
        image = decode_image(payload)
        options = {}
        for name, shapes in (("intrinsics", ((3, 3),)), ("extrinsics", ((3, 4), (4, 4)))):
            value = payload.get(name)
            if value is not None:
                value = np.asarray(value, dtype=np.float32)
                if value.shape not in shapes or not np.isfinite(value).all():
                    raise ValueError(f"Invalid {name} matrix")
            options[name] = value
        focal = payload.get("focal")
        if focal is not None and (not np.isfinite(float(focal)) or float(focal) <= 0):
            raise ValueError("focal must be positive")
        use_ray_pose = payload.get("use_ray_pose", False)
        if not isinstance(use_ray_pose, bool):
            raise ValueError("use_ray_pose must be a boolean")
        resolution = int(payload.get("process_res", 504))
        if resolution <= 0:
            raise ValueError("process_res must be positive")
        method = payload.get("process_res_method", "upper_bound_resize")
        if method not in {"upper_bound_resize", "lower_bound_resize"}:
            raise ValueError("process_res_method must preserve the full image: upper_bound_resize or lower_bound_resize")
        return predictor(image, **options, focal=focal, use_ray_pose=use_ray_pose,
                         process_res=resolution,
                         process_res_method=method)
    return create_app("depth_anything", {"/da3_query": query})


def main():
    cli = parser(__doc__, 4003)
    cli.add_argument("--model", default=DEFAULT_MODEL)
    cli.add_argument("--device", default="cuda")
    args = cli.parse_args()
    serve(build_app(DepthAnything(args.model, args.device)), args)


if __name__ == "__main__":
    main()
