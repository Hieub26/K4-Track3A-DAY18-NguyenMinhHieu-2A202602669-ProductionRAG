from __future__ import annotations

"""Shared Gemini client (OpenAI-compatible endpoint) — throttle + retry cho free tier."""

import os, sys, re, time, threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY, GEMINI_BASE_URL, GEMINI_MODEL, GEMINI_RPM

_client = None
_lock = threading.Lock()
_last_call = 0.0


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        # max_retries=0: tự retry bên dưới để tôn trọng "Please retry in Xs" của Gemini.
        _client = OpenAI(api_key=GEMINI_API_KEY, base_url=GEMINI_BASE_URL, max_retries=0, timeout=60)
    return _client


def _throttle() -> None:
    """Giãn các request để không vượt GEMINI_RPM (free tier giới hạn request/phút)."""
    global _last_call
    with _lock:
        wait = _last_call + 60.0 / GEMINI_RPM - time.time()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.time()


def chat(messages: list[dict], json_mode: bool = False, max_retries: int = 6) -> str | None:
    """Gọi Gemini chat completion. Trả về None nếu không có key; raise nếu hết retry."""
    if not GEMINI_API_KEY:
        return None
    kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
    for attempt in range(max_retries + 1):
        _throttle()
        try:
            # reasoning_effort="low": Gemini 3 không tắt được thinking, để mặc định thì chậm và tốn token.
            resp = _get_client().chat.completions.create(
                model=GEMINI_MODEL, messages=messages, reasoning_effort="low", **kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            status = getattr(e, "status_code", None)
            # Chỉ retry lỗi tạm thời: 429 (quota/phút), 5xx (quá tải), timeout/mất kết nối (status None).
            if attempt == max_retries or (status is not None and status != 429 and status < 500):
                raise
            # Hết quota theo NGÀY: Gemini báo "retry in 14h27m56s" → chờ vô ích, fail ngay để caller fallback.
            if re.search(r"retry in \d+h", str(e)):
                raise
            hinted = re.search(r"retry in (?:(\d+)m)?([\d.]+)s", str(e))
            delay = (int(hinted.group(1) or 0) * 60 + float(hinted.group(2)) + 1 if hinted
                     else min(5 * 2 ** attempt, 60))
            print(f"  ⏳ Gemini {status or type(e).__name__} — thử lại sau {delay:.0f}s", flush=True)
            time.sleep(delay)
    return None


def parse_json(raw: str | None) -> dict:
    """Parse JSON từ output LLM (chịu được ```json fences). Trả về {} nếu hỏng."""
    import json
    if not raw:
        return {}
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
    return data if isinstance(data, dict) else {}
