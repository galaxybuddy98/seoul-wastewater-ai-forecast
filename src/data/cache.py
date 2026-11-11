"""원본 API 페이지와 재개 체크포인트를 JSON으로 보존한다."""

import json
import os
from pathlib import Path


class RawPageCache:
    def __init__(self, root):
        self.root = Path(root)

    def _service_dir(self, service_name):
        path = self.root / service_name
        path.mkdir(parents=True, exist_ok=True)
        return path

    def page_path(self, service_name, start, end):
        return self._service_dir(service_name) / f"{start:09d}_{end:09d}.json"

    def save_page(self, service_name, start, end, payload):
        path = self.page_path(service_name, start, end)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temporary, path)
        return path

    def load_page(self, service_name, start, end):
        path = self.page_path(service_name, start, end)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def save_checkpoint(self, service_name, next_start):
        path = self._service_dir(service_name) / "checkpoint.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"next_start": next_start}), encoding="utf-8")
        os.replace(temporary, path)
        return path

    def load_checkpoint(self, service_name):
        path = self._service_dir(service_name) / "checkpoint.json"
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
