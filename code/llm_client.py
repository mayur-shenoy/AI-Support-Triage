from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic
from openai import OpenAI


@dataclass(slots=True)
class LLMConfig:
    provider: str
    model: str
    api_key: str
    base_url: str


class LLMClient:
    def __init__(self) -> None:
        self.config = self._discover_config()
        self._openai_client = self._build_openai_client()
        self._anthropic_client = self._build_anthropic_client()

    @property
    def enabled(self) -> bool:
        return self.config is not None

    def generate_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        if not self.config:
            return None
        if self.config.provider in {"groq", "openai"}:
            return self._openai_compatible_json(system_prompt, user_prompt)
        if self.config.provider == "anthropic":
            return self._anthropic_json(system_prompt, user_prompt)
        return None

    def generate_text(self, system_prompt: str, user_prompt: str) -> str | None:
        if not self.config:
            return None
        if self.config.provider in {"groq", "openai"}:
            return self._openai_compatible_text(system_prompt, user_prompt)
        if self.config.provider == "anthropic":
            return self._anthropic_text(system_prompt, user_prompt)
        return None

    def _openai_compatible_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            content = response.choices[0].message.content
        except Exception:
            return None
        if not content:
            return None
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return None

    def _anthropic_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any] | None:
        try:
            body = self._anthropic_client.messages.create(
                model=self.config.model,
                max_tokens=800,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception:
            return None

        text_parts = [part.text for part in body.content if getattr(part, "type", None) == "text"]
        joined = "\n".join(text_parts)
        try:
            return json.loads(joined)
        except json.JSONDecodeError:
            return None

    def _openai_compatible_text(self, system_prompt: str, user_prompt: str) -> str | None:
        try:
            response = self._openai_client.chat.completions.create(
                model=self.config.model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception:
            return None
        content = response.choices[0].message.content
        return content.strip() if content else None

    def _anthropic_text(self, system_prompt: str, user_prompt: str) -> str | None:
        try:
            body = self._anthropic_client.messages.create(
                model=self.config.model,
                max_tokens=800,
                temperature=0,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except Exception:
            return None
        text_parts = [part.text for part in body.content if getattr(part, "type", None) == "text"]
        return "\n".join(text_parts).strip() or None

    def _build_openai_client(self) -> OpenAI | None:
        if not self.config or self.config.provider not in {"groq", "openai"}:
            return None
        return OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def _build_anthropic_client(self) -> Anthropic | None:
        if not self.config or self.config.provider != "anthropic":
            return None
        return Anthropic(api_key=self.config.api_key)

    @staticmethod
    def _discover_config() -> LLMConfig | None:
        provider = os.getenv("ORCHESTRATE_LLM_PROVIDER", "").strip().lower()
        if provider == "groq" and os.getenv("GROQ_API_KEY"):
            return LLMConfig(
                provider="groq",
                model=os.getenv("ORCHESTRATE_LLM_MODEL", "llama-3.3-70b-versatile"),
                api_key=os.environ["GROQ_API_KEY"],
                base_url="https://api.groq.com/openai/v1",
            )
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            return LLMConfig(
                provider="openai",
                model=os.getenv("ORCHESTRATE_LLM_MODEL", "gpt-4o-mini"),
                api_key=os.environ["OPENAI_API_KEY"],
                base_url="https://api.openai.com/v1",
            )
        if provider == "anthropic" and os.getenv("ANTHROPIC_API_KEY"):
            return LLMConfig(
                provider="anthropic",
                model=os.getenv("ORCHESTRATE_LLM_MODEL", "claude-3-5-haiku-latest"),
                api_key=os.environ["ANTHROPIC_API_KEY"],
                base_url="https://api.anthropic.com/v1",
            )
        return None
