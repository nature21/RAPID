"""Addresses of locally hosted or remote reconstruction services."""

import os


DEFAULT_PORTS = {"sam3d": 4001, "sam3": 4002, "depth_anything": 4003, "foundation_pose": 4004}


def get_endpoint(service: str) -> tuple[str, int]:
    if service not in DEFAULT_PORTS:
        raise ValueError(f"Unknown VFM service {service!r}")
    prefix = f"RAPID_{service.upper()}"
    host = os.environ.get(f"{prefix}_HOST", "127.0.0.1").strip()
    if not host or "://" in host or "/" in host:
        raise ValueError(f"{prefix}_HOST must be a hostname or IP address")
    try:
        port = int(os.environ.get(f"{prefix}_PORT", DEFAULT_PORTS[service]))
    except ValueError as exc:
        raise ValueError(f"{prefix}_PORT must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"{prefix}_PORT must be between 1 and 65535")
    return host, port
