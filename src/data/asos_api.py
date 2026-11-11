"""기상청 ASOS 일자료 요청 인자."""

ASOS_DAILY_URL = "https://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"


def build_asos_params(api_key, start_date, end_date, station_id, page_no=1):
    if not api_key:
        raise ValueError("공공데이터포털 인증키가 필요합니다")
    return {"serviceKey": api_key, "pageNo": page_no, "numOfRows": 999,
            "dataType": "JSON", "dataCd": "ASOS", "dateCd": "DAY",
            "startDt": start_date, "endDt": end_date, "stnIds": str(station_id)}
