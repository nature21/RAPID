import base64

import numpy as np
from PIL import Image

from real2sim.utils import convert_pil_image_to_base64
from real2sim.vfm_clients.config import get_endpoint
from real2sim.vfm_clients.http import post_json


DEFAULT_HOST, DEFAULT_PORT = get_endpoint("sam3")
DEFAULT_SERVER_URL = f"http://{DEFAULT_HOST}:{DEFAULT_PORT}/sam3_query"


def process_return_json(response, image_size):
    scores = response["scores"]
    masks = response["masks"]
    bboxes = response["bboxes"]
    decode_masks = []

    mask_num = len(scores)
    for i in range(mask_num):
        mask_base64 = masks[i]
        mask_byte = base64.b64decode(mask_base64)
        decode_mask = np.frombuffer(mask_byte, dtype=np.uint8).reshape(
            image_size[1],
            image_size[0],
        )
        decode_masks.append(decode_mask)

    return scores, bboxes, decode_masks


def sam3_request(
    image: Image,
    text: str = None,
    bbox: np.array = None,
    confidence_threshold=0.1,
    server_url=DEFAULT_SERVER_URL,
):
    """
    rgb_image: PIL.image
    text: str
    bbox:  np.array([[x_1, y_1, x_2, y_2]])
    """
    image_base64 = convert_pil_image_to_base64(image)

    payload = {
        "rgb_image": image_base64,
        "text": text,
        "bbox": bbox,
        "confidence_threshold": confidence_threshold,
    }

    for key, value in payload.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.tolist()

    return process_return_json(post_json(server_url, payload), image.size)
