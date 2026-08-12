"""Minimal DeepSeek chat-completions client without an SDK dependency."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any
from urllib import error, request

DEEPSEEK_CHAT_COMPLETIONS_URL = "https://api.deepseek.com/v1/chat/completions"


@dataclass(slots=True)
class DeepSeekChatClient:
    api_key: str
    model: str = "deepseek-chat"
    timeout: int = 30
    max_retries: int = 1
    url: str = DEEPSEEK_CHAT_COMPLETIONS_URL

    def __post_init__(self) -> None:
        if not str(self.api_key).strip():
            raise ValueError("DeepSeek api_key must not be empty")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 1200,
        response_format: dict[str, str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                req = request.Request(self.url, data=body, headers=headers, method="POST")
                with request.urlopen(req, timeout=self.timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                content = str(data["choices"][0]["message"]["content"]).strip()
                if not content:
                    raise ValueError("DeepSeek returned empty content")
                return content
            except (error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
                last_error = exc
        raise RuntimeError("DeepSeek chat request failed") from last_error
