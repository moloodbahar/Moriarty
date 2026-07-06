"""
gemini_client.py — Minimal Gemini wrapper with the same interface as
judges.LLMClient (json_call + CallLogger), so the predictor can treat
same-model and cross-model configs identically.

Requires: pip install google-generativeai
Env: GEMINI_API_KEY
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

from judges import CallLogger, utc_now


def _extract_json(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            raise
        return json.loads(m.group(0))


class GeminiClient:
    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        api_key: Optional[str] = None,
        max_retries: int = 3,
        retry_sleep: float = 2.0,
        log_path: Optional[str] = None,
    ):
        import google.generativeai as genai
        genai.configure(api_key=api_key or os.environ.get("GEMINI_API_KEY"))
        self.model = model
        self._genai = genai
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.logger = CallLogger(log_path)

    def json_call(self, system: str, user: str, temperature: float = 0.0,
                  purpose: str = "unspecified") -> dict:
        last_err: Optional[Exception] = None
        model = self._genai.GenerativeModel(
            self.model,
            system_instruction=system,
            generation_config={
                "temperature": temperature,
                "response_mime_type": "application/json",
            },
        )
        for attempt in range(self.max_retries):
            try:
                resp = model.generate_content(user)
                raw = resp.text
                out = _extract_json(raw)
                out["_raw"] = raw
                self.logger.log({
                    "purpose": purpose, "model": self.model,
                    "temperature": temperature, "attempt": attempt,
                    "system": system, "user": user,
                    "response_raw": raw, "ok": True,
                })
                return out
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(self.retry_sleep * (attempt + 1))
        self.logger.log({
            "purpose": purpose, "model": self.model, "temperature": temperature,
            "system": system, "user": user, "ok": False, "error": str(last_err),
        })
        raise RuntimeError(f"gemini json_call failed after {self.max_retries}: {last_err}")


def make_client(model: str, log_path: Optional[str] = None):
    """Factory: OpenAI-backed or Gemini-backed client with the shared
    json_call interface, chosen by model-name prefix."""
    if model.startswith("gemini"):
        return GeminiClient(model=model, log_path=log_path)
    from judges import LLMClient
    return LLMClient(model=model, log_path=log_path)