from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def get_json(url: str, params: dict[str, object] | None = None, timeout: float = 15) -> Any:
    if params:
        url = f"{url}?{urlencode(params)}"
    request = Request(url, headers={"Accept": "application/json", "User-Agent": "prediction-agent/0.1"})
    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def post_json(
    url: str,
    payload: dict[str, object],
    *,
    headers: dict[str, str] | None = None,
    timeout: float = 15,
    attempts: int = 3,
) -> Any:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request_headers = {"Content-Type": "application/json; charset=utf-8", "User-Agent": "prediction-agent/0.1"}
    request_headers.update(headers or {})
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, data=body, headers=request_headers, method="POST")
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # network/HTTP errors are retried, then surfaced
            error = exc
            if attempt + 1 < attempts:
                time.sleep(2 ** attempt)
    assert error is not None
    raise error
