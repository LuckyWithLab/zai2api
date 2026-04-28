from typing import Any, Dict, Optional, Tuple

upstream_cooldown_until: float = 0.0


def upstream_sse_error(z_data: Dict[str, Any]) -> Optional[Tuple[int, str]]:
    error = z_data.get("error") if isinstance(z_data.get("error"), dict) else None
    if not error:
        return None
    code = str(error.get("code") or "").upper()
    detail = str(error.get("detail") or error.get("message") or code or "Upstream request failed")
    if code == "MODEL_CONCURRENCY_LIMIT":
        return 429, detail
    return 502, detail


def should_retry_upstream_error(status_code: int, detail: str) -> bool:
    if status_code in {429, 500, 502, 503, 504}:
        return True
    if status_code != 405:
        return False
    normalized = detail.lower()
    return any(
        marker in normalized
        for marker in (
            "blocked as it may cause potential threats",
            "model_concurrency_limit",
            "internal server error",
            "oops, something went wrong",
        )
    )


def upstream_cooldown_seconds(status_code: int, detail: str, attempt: int) -> float:
    normalized = detail.lower()
    if status_code == 405:
        return min(30.0 * (attempt + 1), 90.0)
    if "model_concurrency_limit" in normalized:
        return min(10.0 * (attempt + 1), 30.0)
    if status_code in {429, 500, 502, 503, 504}:
        return min(3.0 * (attempt + 1), 15.0)
    return 0.0
