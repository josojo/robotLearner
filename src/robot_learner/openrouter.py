"""OpenRouter-backed language model integration."""

import os

from openrouter import OpenRouter, components


class OpenRouterLanguageModel:
    """Small adapter around the official OpenRouter Python SDK."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str = "openai/gpt-4o-mini",
        client: OpenRouter | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("an OpenRouter API key is required")
        if not model:
            raise ValueError("an OpenRouter model is required")
        self._model = model
        self._client = client or OpenRouter(
            api_key=api_key,
            x_open_router_title="Robot Learner",
        )

    @classmethod
    def from_env(cls) -> "OpenRouterLanguageModel":
        """Build a client from the standard OpenRouter environment variables."""
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )

    def complete(self, prompt: str, *, system_prompt: str | None = None) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages: list[components.ChatMessages] = []
        if system_prompt is not None:
            messages.append(components.ChatSystemMessage(role="system", content=system_prompt))
        messages.append(components.ChatUserMessage(role="user", content=prompt))

        response = self._client.chat.send(
            model=self._model,
            messages=messages,
            stream=False,
        )
        if not response.choices:
            raise RuntimeError("OpenRouter returned no completion choices")
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content:
            raise RuntimeError("OpenRouter returned an empty text completion")
        return content
