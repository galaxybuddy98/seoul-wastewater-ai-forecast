import pytest

from src.data.seoul_api import SeoulApiError, parse_page, page_ranges


def test_page_ranges_use_1000_row_inclusive_pages():
    assert list(page_ranges(total_count=2501)) == [(1, 1000), (1001, 2000), (2001, 2501)]


def test_parse_page_reads_rows_and_total_count():
    payload = {
        "ViewTreQty": {
            "list_total_count": 1,
            "RESULT": {"CODE": "INFO-000", "MESSAGE": "정상 처리되었습니다"},
            "row": [{"MSRMT_YMD": "2026/01/01", "WRCT_NM": "탄천물재생센터"}],
        }
    }

    page = parse_page(payload, "ViewTreQty")

    assert page.total_count == 1
    assert page.rows[0]["WRCT_NM"] == "탄천물재생센터"


def test_parse_page_rejects_api_error_inside_http_success():
    payload = {"RESULT": {"CODE": "INFO-100", "MESSAGE": "인증키가 유효하지 않습니다"}}

    with pytest.raises(SeoulApiError, match="INFO-100"):
        parse_page(payload, "ViewQowRep")
