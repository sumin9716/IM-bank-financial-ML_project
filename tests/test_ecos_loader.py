import pytest
import requests

from fx_external_pipeline_full import ecos_loader


def _failing_get(url, timeout=None):
    response = requests.Response()
    response.status_code = 500
    response.reason = "Internal Server Error"
    response.url = url
    return response


def test_http_error_message_hides_api_key(monkeypatch):
    monkeypatch.setenv("ECOS_API_KEY", "ecos-secret-456")
    monkeypatch.setattr(ecos_loader.requests, "get", _failing_get)
    with pytest.raises(requests.HTTPError) as excinfo:
        ecos_loader.load_ecos_series("722Y001", "M", "2024-01", "2024-03")
    assert "ecos-secret-456" not in str(excinfo.value)
    assert "/StatisticSearch/***/json/" in str(excinfo.value)
