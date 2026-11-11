"""공공 API 공통 HTTP 세션."""

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry


def build_retrying_session() -> requests.Session:
    retry = Retry(total=4, connect=4, read=4, status=4, backoff_factor=0.5,
                  status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=frozenset({"GET"}), respect_retry_after_header=True)
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session
