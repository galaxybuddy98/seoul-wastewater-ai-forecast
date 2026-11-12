"""서울 열린데이터광장 페이지 수집 공개 인터페이스."""

from dataclasses import dataclass

from src.data.api_client import build_retrying_session


class SeoulApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class SeoulPage:
    total_count: int
    rows: list[dict]


def page_ranges(total_count, page_size=1000):
    if total_count < 0 or not 1 <= page_size <= 1000:
        raise ValueError("페이지 범위가 유효하지 않습니다")
    for start in range(1, total_count + 1, page_size):
        yield start, min(start + page_size - 1, total_count)


def parse_page(payload, service_name):
    body = payload.get(service_name)
    if body is None:
        result = payload.get("RESULT", {})
        raise SeoulApiError(f"{result.get('CODE', 'UNKNOWN')}: {result.get('MESSAGE', '응답 없음')}")
    result = body.get("RESULT", {})
    if result.get("CODE") != "INFO-000":
        raise SeoulApiError(f"{result.get('CODE', 'UNKNOWN')}: {result.get('MESSAGE', 'API 오류')}")
    rows = body.get("row") or []
    if isinstance(rows, dict):
        rows = [rows]
    return SeoulPage(int(body.get("list_total_count", len(rows))), list(rows))


class SeoulApiClient:
    def __init__(self, api_key, session=None, timeout=(3.05, 30.0)):
        if not api_key:
            raise ValueError("서울 API 인증키가 필요합니다")
        self._api_key = api_key
        self._session = session or build_retrying_session()
        self._timeout = timeout
        self._base_url = "https://openapi.seoul.go.kr"

    def __repr__(self):
        return "SeoulApiClient(api_key=<비공개>)"

    def fetch_page(self, service_name, start, end):
        if end - start + 1 > 1000:
            raise ValueError("서울 API는 요청당 최대 1,000건입니다")
        url = f"{self._base_url}/{self._api_key}/json/{service_name}/{start}/{end}/"
        try:
            response = self._session.get(url, timeout=self._timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            raise SeoulApiError(f"{service_name} 요청 실패(인증키 비공개)") from None
        return parse_page(payload, service_name)

    def iter_rows(self, service_name):
        first = self.fetch_page(service_name, 1, 1000)
        yield from first.rows
        for start, end in page_ranges(first.total_count):
            if start != 1:
                yield from self.fetch_page(service_name, start, end).rows
