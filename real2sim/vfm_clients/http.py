"""Bounded HTTP requests with actionable model-service errors."""

import requests


def post_json(url, payload, timeout=600):
    try:
        response = requests.post(url, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Model service request to {url} failed: {exc}") from exc
    try:
        result = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Model service at {url} returned non-JSON (HTTP {response.status_code})") from exc
    if not response.ok or not isinstance(result, dict) or "error" in result:
        error = result.get("error", result) if isinstance(result, dict) else result
        raise RuntimeError(f"Model service at {url} failed (HTTP {response.status_code}): {error}")
    return result
