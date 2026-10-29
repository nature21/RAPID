import base64
from io import BytesIO

import numpy as np
import trimesh

from real2sim.utils import convert_pil_image_to_base64
from real2sim.vfm_clients.config import get_endpoint
from real2sim.vfm_clients.http import post_json


DEFAULT_HOST, DEFAULT_PORT = get_endpoint("sam3d")
DEFAULT_SERVER_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/sam3d_query"


def sam3d_request(image, mask, point_map=None, server_url=DEFAULT_SERVER_URL):
    image_base64 = convert_pil_image_to_base64(image)
    mask_bytes = np.asarray(mask, dtype=np.uint8).tobytes()
    mask_base64 = base64.b64encode(mask_bytes).decode("utf-8")
    payload = {
        "rgb_image": image_base64,
        "mask": mask_base64,
    }
    if point_map is not None:
        point_map_bytes = point_map.astype(np.float32).tobytes()
        point_map_base64 = base64.b64encode(point_map_bytes).decode("utf-8")
        payload["point_map"] = point_map_base64

    response_json = post_json(server_url, payload)

    glb_bytes = base64.b64decode(response_json["glb"])
    glb_trimesh = trimesh.load_mesh(BytesIO(glb_bytes), file_type="glb", process=False)

    scale = np.array(response_json["scale"])
    translation = np.array(response_json["translation"])
    quat = np.array(response_json["quat"])

    glb_bytes_processed = base64.b64decode(response_json["glb_processed"])
    glb_trimesh_processed = trimesh.load_mesh(
        BytesIO(glb_bytes_processed),
        file_type="glb",
        process=False,
    )

    return glb_trimesh, glb_trimesh_processed, scale, translation, quat
