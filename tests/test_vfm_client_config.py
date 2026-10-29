import pytest

from real2sim.vfm_clients.config import get_endpoint


def test_local_defaults(monkeypatch):
    for service, port in (("sam3d", 4001), ("sam3", 4002), ("depth_anything", 4003),
                          ("foundation_pose", 4004)):
        monkeypatch.delenv(f"RAPID_{service.upper()}_HOST", raising=False)
        monkeypatch.delenv(f"RAPID_{service.upper()}_PORT", raising=False)
        assert get_endpoint(service) == ("127.0.0.1", port)


def test_remote_endpoint(monkeypatch):
    monkeypatch.setenv("RAPID_SAM3_HOST", "gpu.example.org")
    monkeypatch.setenv("RAPID_SAM3_PORT", "8402")
    assert get_endpoint("sam3") == ("gpu.example.org", 8402)


@pytest.mark.parametrize("name,value", [("HOST", "http://localhost"), ("HOST", ""),
                                       ("PORT", "0"), ("PORT", "65536"), ("PORT", "bad")])
def test_invalid_endpoint(monkeypatch, name, value):
    monkeypatch.setenv(f"RAPID_SAM3_{name}", value)
    with pytest.raises(ValueError, match=f"RAPID_SAM3_{name}"):
        get_endpoint("sam3")
