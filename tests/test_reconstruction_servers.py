"""Exercise production HTTP adapters with deterministic model substitutes."""

import base64
import sys
from contextlib import contextmanager
from threading import Thread
from types import ModuleType, SimpleNamespace

import mujoco
import numpy as np
from PIL import Image
import pytest
import requests
import trimesh
from werkzeug.serving import make_server

from real2sim.pipeline import run_pipeline
from real2sim.mesh_utils import to_trimesh
from real2sim.schema import PipelineConfig
from real2sim.utils import convert_pil_image_to_base64
from real2sim.vfm_clients.depth_anything import decode_array, depth_anything_request
from real2sim.vfm_clients.foundation_pose import fp_registration_request, fp_tracking_request
from real2sim.vfm_clients.sam3 import sam3_request
from real2sim.vfm_clients.sam3d import sam3d_request
from real2sim.workflow import package_reconstruction
from server import depth_anything_server, foundation_pose_server, sam3_server, sam3d_server
from server.common import encode_array, encode_bytes


@contextmanager
def running(app):
    httpd = make_server("127.0.0.1", 0, app)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}"
    finally:
        httpd.shutdown()
        thread.join(timeout=5)
        httpd.server_close()
        assert not thread.is_alive()


@pytest.fixture
def image():
    rgb = np.zeros((48, 80, 3), dtype=np.uint8)
    rgb[12:36, 20:60] = (240, 20, 40)
    return Image.fromarray(rgb)


@pytest.fixture
def services():
    calls = []
    mesh = trimesh.creation.box(extents=(0.04, 0.03, 0.02))
    glb = base64.b64encode(mesh.export(file_type="glb")).decode("ascii")

    def segment(image, text, bbox, confidence):
        calls.append(("sam3", image.size, text, confidence))
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[12:36, 20:60] = 1
        return {"scores": [0.9], "bboxes": [[20, 12, 60, 36]], "masks": [encode_bytes(mask)]}

    def reconstruct(image, mask, pointmap):
        calls.append(("sam3d", mask.copy(), None if pointmap is None else pointmap.copy()))
        assert mask.shape == (image.height, image.width)
        return {"glb": glb, "glb_processed": glb, "scale": [1, 1, 1],
                "translation": [0, 0, 0], "quat": [1, 0, 0, 0]}

    def register(mesh, K, color, depth, **kwargs):
        calls.append(("foundation_pose", kwargs))
        assert color.shape[:2] == depth.shape == (48, 80)
        assert mesh.vertices.shape[1] == 3
        pose = np.eye(4, dtype=np.float32)
        pose[2, 3] = 0.01
        return {"pose": pose.tolist(), "pose_last": pose.tolist()}

    def depth(image, **options):
        calls.append(("depth_anything", options))
        K = np.array([[100, 0, 40], [0, 100, 24], [0, 0, 1]], dtype=np.float32)
        return {"depth": encode_array(np.ones((image.height, image.width), dtype=np.float32)),
                "intrinsics": encode_array(K), "extrinsics": encode_array(np.eye(4, dtype=np.float32)[:3]),
                "is_metric": 1, "units": "meters", "extrinsics_convention": "opencv_w2c_3x4"}

    with (running(sam3_server.build_app(segment)) as sam3,
          running(sam3d_server.build_app(reconstruct)) as sam3d,
          running(foundation_pose_server.build_app(register)) as fp,
          running(depth_anything_server.build_app(depth)) as da3):
        yield SimpleNamespace(sam3=sam3, sam3d=sam3d, fp=fp, da3=da3, calls=calls, mesh=mesh)


@pytest.mark.parametrize("device", ["cpu", "cuda", "cuda:1"])
def test_sam3_adapter_uses_packaged_vocabulary_and_selected_device(monkeypatch, image, device):
    import torch

    model = object()
    mask = torch.zeros((1, 1, image.height, image.width), dtype=torch.bool)
    mask[:, :, 12:36, 20:60] = True
    prediction = {"scores": torch.tensor([0.9]),
                  "boxes": torch.tensor([[20, 12, 60, 36]]), "masks": mask}
    calls = []

    def build_model(*, bpe_path=None, checkpoint_path=None, device="cuda",
                    enable_inst_interactivity=False):
        assert bpe_path is None
        assert checkpoint_path == "checkpoint.pt"
        assert enable_inst_interactivity
        calls.append(("model", device))
        return model

    class Processor:
        def __init__(self, supplied_model, confidence_threshold=0.5, device="cuda"):
            assert supplied_model is model
            assert confidence_threshold == 0.2
            calls.append(("processor", device))

        def set_image(self, supplied_image):
            assert supplied_image.size == image.size
            return prediction

        def reset_all_prompts(self, state):
            assert state is prediction

        def set_text_prompt(self, *, state, prompt):
            assert prompt == "box"
            return state

    sam3 = ModuleType("sam3")
    sam3.build_sam3_image_model = build_model
    processor_module = ModuleType("sam3.model.sam3_image_processor")
    processor_module.Sam3Processor = Processor
    for name, module in {"sam3": sam3, "sam3.model": ModuleType("sam3.model"),
                         "sam3.model.sam3_image_processor": processor_module}.items():
        monkeypatch.setitem(sys.modules, name, module)

    backend = sam3_server.SAM3(checkpoint="checkpoint.pt", device=device)
    client = sam3_server.build_app(backend).test_client()
    response = client.post("/sam3_query", json={
        "rgb_image": convert_pil_image_to_base64(image), "text": "box",
        "confidence_threshold": 0.2,
    })
    assert response.status_code == 200
    assert calls == [("model", device), ("processor", device)]
    result = response.get_json()
    assert result["scores"] == pytest.approx([0.9])
    assert result["bboxes"] == [[20, 12, 60, 36]]
    decoded = np.frombuffer(base64.b64decode(result["masks"][0]), dtype=np.uint8)
    np.testing.assert_array_equal(decoded.reshape(image.height, image.width), mask[0, 0].numpy())


def test_real_clients_roundtrip_non_square_inputs(services, image):
    for url in (services.sam3, services.sam3d, services.fp, services.da3):
        assert requests.get(url + "/health", timeout=5).json()["status"] == "ok"
    scores, boxes, masks = sam3_request(image, "box", server_url=services.sam3 + "/sam3_query")
    assert scores == [0.9]
    assert boxes == [[20, 12, 60, 36]]
    assert masks[0].shape == (48, 80) and masks[0].sum() == 24 * 40
    pointmap = np.arange(48 * 80 * 3, dtype=np.float32).reshape(48, 80, 3)
    mesh, processed, scale, translation, quat = sam3d_request(
        image, masks[0].astype(bool), pointmap, server_url=services.sam3d + "/sam3d_query")
    np.testing.assert_array_equal(services.calls[-1][2], pointmap)
    np.testing.assert_array_equal(scale, [1, 1, 1])
    np.testing.assert_allclose(to_trimesh(mesh).bounds, services.mesh.bounds)
    np.testing.assert_allclose(to_trimesh(processed).bounds, services.mesh.bounds)
    depth, metadata = depth_anything_request(image, return_metadata=True,
                                            server_url=services.da3 + "/da3_query")
    assert depth.shape == (48, 80) and depth.dtype == np.float32
    pose, state = fp_registration_request(services.mesh, metadata["intrinsics"], np.asarray(image),
                                         depth, masks[0], server_url=services.fp + "/fp_registration_query")
    tracked, _ = fp_tracking_request(state, services.mesh, metadata["intrinsics"], np.asarray(image),
                                     depth, server_url=services.fp + "/fp_tracking_query")
    np.testing.assert_allclose(pose, tracked)
    np.testing.assert_array_equal(services.calls[-1][1]["pose_last"], state)


@pytest.mark.parametrize("module,route", [
    (sam3_server, "/sam3_query"), (sam3d_server, "/sam3d_query"),
    (foundation_pose_server, "/fp_registration_query"),
    (depth_anything_server, "/da3_query"),
])
def test_missing_fields_do_not_call_model(module, route):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid input reached model")
    client = module.build_app(forbidden).test_client()
    assert client.post(route, json={}).status_code == 400
    assert client.post(route, json=[]).status_code == 400


def test_invalid_mask_bytes_and_server_errors(image):
    app = sam3d_server.build_app(lambda *args: pytest.fail("Invalid mask reached model"))
    response = app.test_client().post("/sam3d_query", json={
        "rgb_image": convert_pil_image_to_base64(image), "mask": encode_bytes(np.ones((80, 48, 2)))})
    assert response.status_code == 400
    assert "bytes" in response.json["error"]

    def fail(*args):
        raise RuntimeError("checkpoint unavailable")
    with running(sam3_server.build_app(fail)) as url:
        with pytest.raises(RuntimeError, match="checkpoint unavailable"):
            sam3_request(image, "box", server_url=url + "/sam3_query")


@pytest.mark.parametrize("payload", [
    {"rgb_image": "not base64", "text": "box"},
    {"rgb_image": base64.b64encode(b"not an image").decode(), "text": "box"},
])
def test_unreadable_image_is_client_error(payload):
    app = sam3_server.build_app(lambda *args: pytest.fail("Invalid image reached model"))
    response = app.test_client().post("/sam3_query", json=payload)
    assert response.status_code == 400
    assert "error" in response.json


@pytest.mark.parametrize("options", [{"process_res_method": "upper_bound_crop"},
                                     {"focal": -1}, {"use_ray_pose": "false"},
                                     {"intrinsics": [[1, 0], [0, 1]]}])
def test_depth_rejects_unsupported_camera_inputs(image, options):
    app = depth_anything_server.build_app(lambda *args, **kwargs: pytest.fail("Invalid input reached model"))
    response = app.test_client().post("/da3_query", json={
        "rgb_image": convert_pil_image_to_base64(image), **options})
    assert response.status_code == 400


def test_service_timeout_and_non_json_errors(monkeypatch):
    from real2sim.vfm_clients.http import post_json

    def timeout(url, **kwargs):
        assert kwargs["timeout"] == 600
        raise requests.Timeout("timeout")
    monkeypatch.setattr(requests, "post", timeout)
    with pytest.raises(RuntimeError, match="timeout"):
        post_json("http://127.0.0.1/unavailable", {})
    response = requests.Response()
    response.status_code = 502
    response._content = b"Bad gateway"
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: response)
    with pytest.raises(RuntimeError, match="non-JSON.*502"):
        post_json("http://127.0.0.1/unavailable", {})


@pytest.mark.parametrize("tuple_result", [False, True])
def test_foundation_pose_preserves_centered_tracking_frame(tuple_result):
    pose, centered = np.eye(4), np.eye(4)
    pose[0, 3], centered[0, 3] = 0.2, 0.3
    result = (pose, centered) if tuple_result else pose
    response = foundation_pose_server.pose_response(result, SimpleNamespace(pose_last=centered))
    assert response["pose"][0][3] == pytest.approx(0.2)
    assert response["pose_last"][0][3] == pytest.approx(0.3)


def test_depth_resizes_camera_and_metric_units(image):
    prediction = SimpleNamespace(depth=np.ones((1, 24, 40)), is_metric=False,
                                 intrinsics=np.array([[[50, 0, 20], [0, 50, 12], [0, 0, 1]]]),
                                 conf=np.ones((1, 24, 40)), extrinsics=np.eye(4)[None])
    response = depth_anything_server.prediction_response(prediction, image, "da3metric-large", focal=600)
    np.testing.assert_allclose(decode_array(response["depth"]), np.full((48, 80), 2))
    np.testing.assert_allclose(decode_array(response["intrinsics"]), [[100, 0, 40], [0, 100, 24], [0, 0, 1]])
    assert response["units"] == "meters"
    assert decode_array(response["extrinsics"]).shape == (3, 4)
    prediction.is_metric = True
    response = depth_anything_server.prediction_response(prediction, image, "da3metric-large", focal=600)
    np.testing.assert_allclose(decode_array(response["depth"]), np.ones((48, 80)))


@pytest.mark.parametrize("supplied_depth", [False, True])
def test_http_pipeline_exports_and_packages_scene(services, image, tmp_path, monkeypatch, supplied_depth):
    monkeypatch.setattr("real2sim.pipeline.detect_objects", lambda *args, **kwargs: (
        [{"name": "box", "description": "red box", "confidence": "high"}], '{"objects": ["box"]}'))
    config = PipelineConfig(
        depth_server_url=services.da3 + "/da3_query", sam3_server_url=services.sam3 + "/sam3_query",
        sam3d_server_url=services.sam3d + "/sam3d_query", foundation_pose_url=services.fp + "/fp_registration_query",
        world_frame_mode="camera", use_static_object_selection=False, use_physical_parameter_selection=False,
        use_grounding_alias_fallback=False, fallback_without_foundation_pose=False,
        continue_on_object_error=False, reconstruction_workers=1, verbose=False,
    )
    kwargs = {}
    if supplied_depth:
        config.intrinsics = np.array([[100, 0, 40], [0, 100, 24], [0, 0, 1]], dtype=float)
        kwargs["depth"] = np.ones((48, 80), dtype=np.float32)
    result = run_pipeline(image, tmp_path / "reconstruction", config, **kwargs)
    assert len(result.objects) == 1 and not result.skipped_objects
    model = mujoco.MjModel.from_xml_path(str(result.scene_xml))
    assert model.ngeom > 0
    assert any(call[0] == "depth_anything" for call in services.calls) is not supplied_depth
    assert {call[0] for call in services.calls} >= {"sam3", "sam3d", "foundation_pose"}
    packaged = package_reconstruction(result.output_dir, name="test_scene", tasks_root=tmp_path / "tasks")
    assert (packaged.task_dir / "scene.py").is_file()
    final_model = mujoco.MjModel.from_xml_path(str(packaged.task_dir / "bundle/scene.xml"))
    data = mujoco.MjData(final_model)
    mujoco.mj_step(final_model, data, nstep=10)
    assert np.isfinite(data.qpos).all()
