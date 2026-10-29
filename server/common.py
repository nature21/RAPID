"""Shared HTTP and array encoding for reconstruction services."""

import argparse
import base64
from io import BytesIO

from flask import Flask, jsonify, request
import numpy as np
from PIL import Image
from werkzeug.exceptions import HTTPException


def decode_image(payload):
    try:
        with Image.open(BytesIO(base64.b64decode(payload["rgb_image"], validate=True))) as image:
            return image.convert("RGB")
    except OSError as exc:
        raise ValueError("rgb_image must encode a readable image") from exc


def decode_array(value, dtype, shape):
    data = base64.b64decode(value, validate=True)
    expected = int(np.prod(shape)) * np.dtype(dtype).itemsize
    if len(data) != expected:
        raise ValueError(f"Expected {expected} bytes for array with shape {shape}")
    return np.frombuffer(data, dtype=dtype).reshape(shape).copy()


def encode_bytes(array):
    return base64.b64encode(np.ascontiguousarray(array).tobytes()).decode("ascii")


def encode_array(array):
    array = np.ascontiguousarray(array)
    return {"data": encode_bytes(array), "shape": list(array.shape), "dtype": str(array.dtype)}


def create_app(service, routes):
    """Build an app after its model is loaded; health therefore means ready."""
    app = Flask(service)
    app.config["MAX_CONTENT_LENGTH"] = 128 * 1024 * 1024

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.get("/health")
    def health():
        return jsonify(status="ok", service=service)

    def dispatch(handler):
        def view():
            try:
                payload = request.get_json()
                if not isinstance(payload, dict):
                    raise ValueError("Expected a JSON object")
                return jsonify(handler(payload))
            except HTTPException:
                raise
            except (KeyError, ValueError, TypeError) as exc:
                return jsonify(error=str(exc)), 400
            except Exception as exc:
                app.logger.exception("Model request failed")
                return jsonify(error=str(exc)), 500
        return view

    for path, handler in routes.items():
        app.add_url_rule(path, endpoint=path, view_func=dispatch(handler), methods=["POST"])
    return app


def parser(description, port):
    result = argparse.ArgumentParser(description=description)
    result.add_argument("--host", "--ip", default="127.0.0.1")
    result.add_argument("--port", type=int, default=port)
    return result


def serve(app, args):
    # One inference at a time also keeps CUDA rasterizer state on its owner thread.
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False, threaded=False)
