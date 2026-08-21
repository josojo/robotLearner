"""OpenRouter-backed language model integration."""

from __future__ import annotations

import base64
import os
from collections.abc import Sequence
from pathlib import Path

from openrouter import OpenRouter, components

_IMAGE_MEDIA_TYPES = {
    ".gif": "image/gif",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


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
    def from_env(cls) -> OpenRouterLanguageModel:
        """Build a client from the standard OpenRouter environment variables."""
        return cls(
            api_key=os.environ.get("OPENROUTER_API_KEY", ""),
            model=os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini"),
        )

    def complete(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        images: Sequence[str | Path] | None = None,
    ) -> str:
        if not prompt.strip():
            raise ValueError("prompt must not be empty")

        messages: list[components.ChatMessages] = []
        if system_prompt is not None:
            messages.append(components.ChatSystemMessage(role="system", content=system_prompt))
        messages.append(
            components.ChatUserMessage(role="user", content=_user_content(prompt, images))
        )

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


def _user_content(
    prompt: str, images: Sequence[str | Path] | None
) -> str | list[components.ChatContentItems]:
    paths = [Path(item) for item in images or ()]
    if not paths:
        return prompt
    parts: list[components.ChatContentItems] = [
        components.ChatContentText(type="text", text=prompt)
    ]
    for path in paths:
        parts.append(_image_part(path))
        parts.append(
            components.ChatContentText(
                type="text",
                text=f"(image: {path.parent.name}/{path.name})"
                if path.parent.name
                else f"(image: {path.name})",
            )
        )
    return parts


def _image_part(path: Path) -> components.ChatContentImage:
    media_type = _IMAGE_MEDIA_TYPES.get(path.suffix.lower())
    if media_type is None:
        raise ValueError(f"unsupported image type for OpenRouter: {path}")
    if not path.is_file():
        raise ValueError(f"image file does not exist: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return components.ChatContentImage(
        type="image_url",
        image_url=components.ChatContentImageImageURL(url=f"data:{media_type};base64,{encoded}"),
    )
