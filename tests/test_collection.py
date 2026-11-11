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
