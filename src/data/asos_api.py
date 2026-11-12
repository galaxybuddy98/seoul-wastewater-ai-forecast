"""기상청 ASOS 일자료 요청 인자."""

ASOS_DAILY_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
ASOS_HOURLY_URL = "https://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"


def build_asos_params(api_key, start_date, end_date, station_id, page_no=1):
    if not api_key:
        raise ValueError("공공데이터포털 인증키가 필요합니다")
    return {"serviceKey": api_key, "pageNo": page_no, "numOfRows": 999,
            "dataType": "JSON", "dataCd": "ASOS", "dateCd": "DAY",
            "startDt": start_date, "endDt": end_date, "stnIds": str(station_id)}


def build_asos_hourly_params(
    api_key, start_date, end_date, station_id, page_no=1, num_rows=999
):
    """ASOS 시간자료 요청 파라미터를 구성한다."""
    if not api_key:
        raise ValueError("공공데이터포털 인증키가 필요합니다")
    return {
        "serviceKey": api_key,
        "pageNo": page_no,
        "numOfRows": num_rows,
        "dataType": "JSON",
        "dataCd": "ASOS",
        "dateCd": "HR",
        "startDt": start_date,
        "startHh": "00",
        "endDt": end_date,
        "endHh": "23",
        "stnIds": str(station_id),
    }


def parse_asos_payload(payload):
    """ASOS 응답을 일평균기온·일강수량 표로 정규화하고 공란을 결측으로 둔다."""
    import pandas as pd

    response = payload.get("response", {})
    header = response.get("header", {})
    code = str(header.get("resultCode", ""))
    if code not in {"00", "0"}:
        raise ValueError(f"ASOS API 오류 {code}: {header.get('resultMsg', '응답 오류')}")
    body = response.get("body", {})
    items = body.get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    frame = pd.DataFrame(items)
    if frame.empty:
        return int(body.get("totalCount", 0)), pd.DataFrame(
            columns=[
                "date", "station_id", "station_name",
                "mean_temperature_c", "precipitation_mm",
            ]
        )
    result = pd.DataFrame(
        {
            "date": pd.to_datetime(frame["tm"], errors="coerce"),
            "station_id": frame["stnId"].astype(str),
            "station_name": frame["stnNm"].astype(str),
            "mean_temperature_c": pd.to_numeric(frame["avgTa"], errors="coerce"),
            "precipitation_mm": pd.to_numeric(frame["sumRn"], errors="coerce"),
        }
    )
    return int(body.get("totalCount", len(result))), result


def parse_asos_hourly_payload(payload):
    """시간자료의 강수값과 QC 원문을 결측 대치 없이 정규화한다."""
    import pandas as pd

    response = payload.get("response", {})
    header = response.get("header", {})
    code = str(header.get("resultCode", ""))
    if code not in {"00", "0"}:
        raise ValueError(f"ASOS 시간자료 API 오류 {code}: {header.get('resultMsg', '응답 오류')}")
    body = response.get("body", {})
    items = body.get("items", {}).get("item", [])
    if isinstance(items, dict):
        items = [items]
    frame = pd.DataFrame(items)
    if frame.empty:
        return int(body.get("totalCount", 0)), pd.DataFrame(
            columns=[
                "timestamp", "date", "station_id", "station_name",
                "hourly_precipitation_mm", "rainfall_qc_flag",
            ]
        )
    timestamp = pd.to_datetime(frame["tm"], errors="coerce")
    result = pd.DataFrame(
        {
            "timestamp": timestamp,
            # rn은 해당 시각 직전 구간 강수량이므로 00시는 전날 일계에 귀속한다.
            "date": (timestamp - pd.Timedelta(seconds=1)).dt.normalize(),
            "station_id": frame["stnId"].astype(str),
            "station_name": frame["stnNm"].astype(str),
            "hourly_precipitation_mm": pd.to_numeric(frame["rn"], errors="coerce"),
            "rainfall_qc_flag": frame.get("rnQcflg", "").fillna("").astype(str),
        }
    )
    return int(body.get("totalCount", len(result))), result
