import base64

import numpy as np
from PIL import Image

from real2sim.utils import convert_pil_image_to_base64
from real2sim.vfm_clients.config import get_endpoint
from real2sim.vfm_clients.http import post_json


DEFAULT_HOST, DEFAULT_PORT = get_endpoint("foundation_pose")
DEFAULT_REGISTRATION_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/fp_registration_query"
DEFAULT_TRACKING_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/fp_tracking_query"


def fp_registration_request(
    mesh,
    K,
    color,
    depth,
    mask,
    server_url=DEFAULT_REGISTRATION_URL,
):
    mesh_bytes = mesh.export(file_type="glb")
    mesh_base64 = base64.b64encode(mesh_bytes).decode("utf-8")

    image_base64 = convert_pil_image_to_base64(Image.fromarray(color))

    depth_bytes = depth.astype(np.float32).tobytes()
    depth_base64 = base64.b64encode(depth_bytes).decode("utf-8")

    mask_bytes = mask.astype(np.uint8).tobytes()
    mask_base64 = base64.b64encode(mask_bytes).decode("utf-8")

    payload = {
        "mesh_file": mesh_base64,
        "intrinsics": K.tolist(),
        "rgb_image": image_base64,
        "depth": depth_base64,
        "mask": mask_base64,
    }

    response_json = post_json(server_url, payload)

    if "pose" not in response_json or "pose_last" not in response_json:
        raise RuntimeError(
            "Foundation Pose registration response missing 'pose' or 'pose_last'"
        )

    return (
        np.array(response_json["pose"]).astype(np.float32).reshape(4, 4),
        np.array(response_json["pose_last"]).astype(np.float32).reshape(4, 4),
    )


def fp_tracking_request(
    pose_last,
    mesh,
    K,
    color,
    depth,
    server_url=DEFAULT_TRACKING_URL,
):
    pose_last_list = pose_last.tolist()

    mesh_bytes = mesh.export(file_type="glb")
    mesh_base64 = base64.b64encode(mesh_bytes).decode("utf-8")

    image_base64 = convert_pil_image_to_base64(Image.fromarray(color))

    depth_bytes = depth.astype(np.float32).tobytes()
    depth_base64 = base64.b64encode(depth_bytes).decode("utf-8")

    payload = {
        "pose_last": pose_last_list,
        "mesh_file": mesh_base64,
        "intrinsics": K.tolist(),
        "rgb_image": image_base64,
        "depth": depth_base64,
    }

    response_json = post_json(server_url, payload)

    if "pose" not in response_json or "pose_last" not in response_json:
        raise RuntimeError(
            "Foundation Pose tracking response missing 'pose' or 'pose_last'"
        )

    return (
        np.array(response_json["pose"]).astype(np.float32).reshape(4, 4),
        np.array(response_json["pose_last"]).astype(np.float32).reshape(4, 4),
    )
