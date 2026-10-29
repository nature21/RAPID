"""SAM 3D Objects reconstruction service with camera-frame mesh output."""

import base64
from pathlib import Path
import sys

import numpy as np

from server.common import create_app, decode_array, decode_image, parser, serve


class SAM3D:
    def __init__(self, repo, config):
        repo = Path(repo).expanduser().resolve()
        sys.path.insert(0, str(repo / "notebook"))
        from inference import Inference

        config = Path(config).expanduser()
        if not config.is_absolute():
            config = repo / config
        self.inference = Inference(str(config), compile=False)

    def __call__(self, image, mask, pointmap):
        import torch
        from pytorch3d.transforms import quaternion_to_matrix, Transform3d

        kwargs = {} if pointmap is None else {"pointmap": torch.from_numpy(pointmap)}
        with torch.inference_mode():
            output = self.inference(np.asarray(image), mask, seed=42, **kwargs)
            mesh = output["glb"]
            scale, rotation, translation = (output[key] for key in ("scale", "rotation", "translation"))
            processed = mesh.copy()
            yup_to_zup = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float32)
            vertices = torch.as_tensor(processed.vertices.astype(np.float32) @ yup_to_zup,
                                       device=rotation.device)
            transform = Transform3d(dtype=scale.dtype, device=scale.device).scale(scale).rotate(
                quaternion_to_matrix(rotation)).translate(translation)
            processed.vertices = transform.transform_points(vertices[None])[0].cpu().numpy()
            processed.apply_transform(np.diag([-1, 1, -1, 1]))
            return {
                "glb": base64.b64encode(mesh.export(file_type="glb")).decode("ascii"),
                "glb_processed": base64.b64encode(processed.export(file_type="glb")).decode("ascii"),
                "scale": scale[0].tolist(), "translation": translation[0].tolist(),
                "quat": rotation[0].tolist(),
            }


def build_app(predictor):
    def query(payload):
        image = decode_image(payload)
        shape = (image.height, image.width)
        mask = decode_array(payload["mask"], np.uint8, shape)
        if not mask.any():
            raise ValueError("mask must contain foreground pixels")
        pointmap = None
        if "point_map" in payload:
            pointmap = decode_array(payload["point_map"], np.float32, (*shape, 3))
        return predictor(image, (mask != 0).astype(np.uint8), pointmap)
    return create_app("sam3d", {"/sam3d_query": query})


def main():
    cli = parser(__doc__, 4001)
    cli.add_argument("--repo", type=Path, required=True, help="SAM 3D Objects checkout")
    cli.add_argument("--config", default="checkpoints/hf/pipeline.yaml")
    args = cli.parse_args()
    serve(build_app(SAM3D(args.repo, args.config)), args)


if __name__ == "__main__":
    main()
