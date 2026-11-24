import numpy as np
import pandas as pd

from src.data.asos_api import build_asos_hourly_params, parse_asos_hourly_payload
from src.data.asos_hourly_collector import collect_hourly_period, iter_year_ranges
from src.data.cache import RawPageCache
from src.preprocessing.rainfall import reconstruct_daily_rainfall, select_hourly_intervals_for_days


def test_hourly_params_use_hour_contract_without_exposing_key():
    params = build_asos_hourly_params(
        "secret", "20250101", "20250101", station_id="108", page_no=2
    )

    assert params["dateCd"] == "HR"
    assert params["startHh"] == "00"
    assert params["endHh"] == "23"
    assert params["pageNo"] == 2
    assert "secret" not in repr({k: v for k, v in params.items() if k != "serviceKey"})


def test_hourly_parser_preserves_blank_rain_and_qc_flag():
    payload = {
        "response": {
            "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
            "body": {
                "totalCount": 2,
                "items": {
                    "item": [
                        {"tm": "2025-05-14 00:00", "stnId": "108", "stnNm": "서울", "rn": "", "rnQcflg": ""},
                        {"tm": "2025-05-14 01:00", "stnId": "108", "stnNm": "서울", "rn": "0.4", "rnQcflg": "0"},
                    ]
                },
            },
        }
    }

    total, frame = parse_asos_hourly_payload(payload)

    assert total == 2
    assert pd.isna(frame.loc[0, "hourly_precipitation_mm"])
    assert frame.loc[0, "date"] == pd.Timestamp("2025-05-13")
    assert frame.loc[1, "hourly_precipitation_mm"] == 0.4
    assert frame.loc[1, "rainfall_qc_flag"] == "0"


def test_daily_reconstruction_uses_known_daily_and_recovers_only_observed_positive_hourly():
    daily = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "precipitation_mm": [1.0, np.nan, np.nan],
        }
    )
    hourly = pd.DataFrame(
        {
            "timestamp": pd.to_datetime([
                "2024-01-01 01:00", "2024-01-01 02:00",
                "2024-01-02 01:00", "2024-01-02 02:00",
                "2024-01-03 01:00", "2024-01-03 02:00",
            ]),
            "date": pd.to_datetime([
                "2024-01-01", "2024-01-01", "2024-01-02",
                "2024-01-02", "2024-01-03", "2024-01-03",
            ]),
            "hourly_precipitation_mm": [0.4, 0.6, 0.2, 0.3, np.nan, np.nan],
            "rainfall_qc_flag": ["0", "0", "0", "0", "", ""],
        }
    )

    result = reconstruct_daily_rainfall(daily, hourly)

    assert result.loc[0, "rainfall_status"] == "A_일자료_확인"
    assert result.loc[0, "validated_precipitation_mm"] == 1.0
    assert result.loc[1, "rainfall_status"] == "A_시간자료_복원"
    assert result.loc[1, "validated_precipitation_mm"] == 0.5
    assert result.loc[2, "rainfall_status"] == "C_미량강수_또는_미확인"
    assert pd.isna(result.loc[2, "validated_precipitation_mm"])


def test_daily_reconstruction_keeps_qc_error_as_unknown():
    daily = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "precipitation_mm": [np.nan]})
    hourly = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 01:00"]),
            "date": pd.to_datetime(["2024-01-01"]),
            "hourly_precipitation_mm": [1.0],
            "rainfall_qc_flag": ["1"],
        }
    )

    result = reconstruct_daily_rainfall(daily, hourly)

    assert result.loc[0, "rainfall_status"] == "D_관측오류_또는_누락"
    assert pd.isna(result.loc[0, "validated_precipitation_mm"])


def test_daily_reconstruction_confirms_zero_only_when_every_returned_rain_value_is_zero():
    daily = pd.DataFrame({"date": pd.to_datetime(["2024-01-01"]), "precipitation_mm": [np.nan]})
    hourly = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-01 00:00", "2024-01-01 03:00"]),
            "date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "hourly_precipitation_mm": [0.0, 0.0],
            "rainfall_qc_flag": ["0", "0"],
        }
    )

    result = reconstruct_daily_rainfall(daily, hourly)

    assert result.loc[0, "rainfall_status"] == "B_시간자료_무강수_확인"
    assert result.loc[0, "validated_precipitation_mm"] == 0.0


def test_hourly_collector_paginates_and_reuses_raw_cache(tmp_path):
    class Response:
        def __init__(self, payload):
            self.payload = payload

        def raise_for_status(self):
            return None

        def json(self):
            return self.payload

    class Session:
        def __init__(self):
            self.pages = []

        def get(self, _url, params, timeout):
            self.pages.append(params["pageNo"])
            page = params["pageNo"]
            item = {
                "tm": f"2024-01-01 0{page - 1}:00", "stnId": "108",
                "stnNm": "서울", "rn": "0.1", "rnQcflg": "0",
            }
            return Response({
                "response": {
                    "header": {"resultCode": "00", "resultMsg": "NORMAL_SERVICE"},
                    "body": {"totalCount": 2, "items": {"item": [item]}},
                }
            })

    cache = RawPageCache(tmp_path)
    first_session = Session()
    first = collect_hourly_period(
        first_session, cache, "key", "108", "20240101", "20240101", rows_per_page=1
    )
    second_session = Session()
    second = collect_hourly_period(
        second_session, cache, "key", "108", "20240101", "20240101", rows_per_page=1
    )

    assert first_session.pages == [1, 2]
    assert second_session.pages == []
    assert len(first) == len(second) == 2


def test_year_ranges_stop_at_requested_final_date():
    assert iter_year_ranges("20250101", "20260930") == [
        ("20250101", "20251231"),
        ("20260101", "20260930"),
    ]


def test_target_interval_filter_keeps_next_midnight_and_excludes_start_midnight():
    hourly = pd.DataFrame(
        {"timestamp": pd.to_datetime([
            "2024-01-01 00:00", "2024-01-01 01:00",
            "2024-01-02 00:00", "2024-01-02 01:00",
        ])}
    )

    result = select_hourly_intervals_for_days(hourly, "2024-01-01", "2024-01-01")

    assert result["timestamp"].tolist() == [
        pd.Timestamp("2024-01-01 01:00"), pd.Timestamp("2024-01-02 00:00")
    ]
