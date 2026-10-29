"""FoundationPose registration and stateless tracking service."""

import base64
from io import BytesIO
from pathlib import Path
import sys
import tempfile

import numpy as np
import trimesh

from server.common import create_app, decode_array, decode_image, parser, serve


def pose_response(result, estimator):
    """Return the mesh pose and the centered-mesh tracking state separately."""
    if isinstance(result, tuple):
        pose, state = result
    else:
        pose, state = result, estimator.pose_last
    if state is None:
        raise RuntimeError("FoundationPose did not establish a tracking state")
    def matrix(value):
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype=np.float32).reshape(4, 4)
        if not np.isfinite(value).all():
            raise RuntimeError("FoundationPose returned a nonfinite pose")
        return value.tolist()
    return {"pose": matrix(pose), "pose_last": matrix(state)}


class FoundationPoseBackend:
    def __init__(self, repo):
        sys.path.insert(0, str(Path(repo).expanduser().resolve()))
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
        import nvdiffrast.torch as dr

        self.estimator_class = FoundationPose
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.context = dr.RasterizeCudaContext()

    def __call__(self, mesh, intrinsics, color, depth, mask=None, pose_last=None):
        import torch

        with tempfile.TemporaryDirectory(prefix="rapid-foundationpose-") as directory:
            estimator = self.estimator_class(
                model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
                scorer=self.scorer, refiner=self.refiner, glctx=self.context,
                debug=0, debug_dir=directory,
            )
            if pose_last is None:
                result = estimator.register(K=intrinsics, rgb=color, depth=depth,
                                            ob_mask=mask.astype(bool), iteration=5)
            else:
                estimator.pose_last = torch.as_tensor(pose_last, dtype=torch.float32, device="cuda")
                result = estimator.track_one(K=intrinsics, rgb=color, depth=depth, iteration=2)
            return pose_response(result, estimator)


def build_app(predictor):
    def query(payload, tracking=False):
        image = decode_image(payload)
        shape = (image.height, image.width)
        depth = decode_array(payload["depth"], np.float32, shape)
        if not np.isfinite(depth).all() or np.any(depth < 0):
            raise ValueError("depth must contain finite, nonnegative metric values")
        intrinsics = np.asarray(payload["intrinsics"], dtype=np.float32)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError("intrinsics must be a finite 3x3 matrix")
        scene = trimesh.load(BytesIO(base64.b64decode(payload["mesh_file"], validate=True)),
                             file_type="glb", process=False)
        mesh = scene.to_mesh() if isinstance(scene, trimesh.Scene) else scene
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices):
            raise ValueError("mesh_file must contain a triangle mesh")
        kwargs = {}
        if tracking:
            pose = np.asarray(payload["pose_last"], dtype=np.float32)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("pose_last must be a finite 4x4 matrix")
            kwargs["pose_last"] = pose
        else:
            mask = decode_array(payload["mask"], np.uint8, shape)
            if not np.any((mask != 0) & (depth > 0)):
                raise ValueError("mask must overlap positive depth")
            kwargs["mask"] = mask
        return predictor(mesh, intrinsics, np.asarray(image), depth, **kwargs)
    return create_app("foundation_pose", {
        "/fp_registration_query": query,
        "/fp_tracking_query": lambda payload: query(payload, tracking=True),
    })


def main():
    cli = parser(__doc__, 4004)
    cli.add_argument("--repo", type=Path, required=True, help="FoundationPose checkout with weights")
    args = cli.parse_args()
    serve(build_app(FoundationPoseBackend(args.repo)), args)


if __name__ == "__main__":
    main()
