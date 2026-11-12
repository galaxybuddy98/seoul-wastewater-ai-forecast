import json

from src.data.asos_api import build_asos_params
from src.data.cache import RawPageCache
from src.data.seoul_api import SeoulApiClient


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self):
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        service = "ViewTreQty"
        start = int(url.rstrip("/").split("/")[-2])
        rows = [{"id": start}]
        return FakeResponse(
            {
                service: {
                    "list_total_count": 1001,
                    "RESULT": {"CODE": "INFO-000", "MESSAGE": "정상"},
                    "row": rows,
                }
            }
        )


def test_seoul_client_fetches_all_inclusive_pages_and_hides_key():
    session = FakeSession()
    client = SeoulApiClient("비밀키", session=session)

    rows = list(client.iter_rows("ViewTreQty"))

    assert [row["id"] for row in rows] == [1, 1001]
    assert len(session.urls) == 2
    assert "비밀키" not in repr(client)


def test_asos_parameters_use_daily_observation_contract():
    params = build_asos_params("키", "20160101", "20160131", station_id="108", page_no=2)

    assert params == {
        "serviceKey": "키",
        "pageNo": 2,
        "numOfRows": 999,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "DAY",
        "startDt": "20160101",
        "endDt": "20160131",
        "stnIds": "108",
    }


def test_raw_page_cache_reuses_page_and_checkpoint(tmp_path):
    cache = RawPageCache(tmp_path)
    payload = {"ViewTreQty": {"row": [{"id": 1}]}}

    path = cache.save_page("ViewTreQty", 1, 1000, payload)
    cache.save_checkpoint("ViewTreQty", next_start=1001)

    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert cache.load_page("ViewTreQty", 1, 1000) == payload
    assert cache.load_checkpoint("ViewTreQty") == {"next_start": 1001}


class FailingSession:
    def get(self, url, timeout):
        raise RuntimeError(f"request failed: {url}")


def test_seoul_client_uses_official_http_endpoint_for_port_8088():
    session = FakeSession()
    client = SeoulApiClient("비밀키", session=session)

    client.fetch_page("ViewTreQty", 1, 1)

    assert session.urls[0].startswith("https://openapi.seoul.go.kr/")


def test_seoul_client_error_does_not_expose_api_key():
    from src.data.seoul_api import SeoulApiError

    client = SeoulApiClient("노출금지키", session=FailingSession())

    try:
        client.fetch_page("ViewTreQty", 1, 1)
    except SeoulApiError as error:
        assert "노출금지키" not in str(error)
    else:
        raise AssertionError("SeoulApiError가 발생해야 합니다")


def test_parse_asos_payload_preserves_blank_precipitation_as_missing():
    import pandas as pd
    from src.data.asos_api import parse_asos_payload

    payload = {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
            "body": {
                "totalCount": 1,
                "items": {
                    "item": [
                        {
                            "tm": "2024-01-01",
                            "stnId": "108",
                            "stnNm": "서울",
                            "avgTa": "1.2",
                            "sumRn": "",
                        }
                    ]
                },
            },
        }
    }

    total, frame = parse_asos_payload(payload)

    assert total == 1
    assert frame.loc[0, "mean_temperature_c"] == 1.2
    assert pd.isna(frame.loc[0, "precipitation_mm"])
