from pathlib import Path
from typing import cast
from unittest.mock import MagicMock

import pytest
from openrouter import OpenRouter, components

from robot_learner.openrouter import OpenRouterLanguageModel
from robot_learner.simulation_testing import MIN_PNG


def test_completion_uses_openrouter_sdk() -> None:
    client = MagicMock()
    client.chat.send.return_value.choices = [
        MagicMock(message=MagicMock(content="observe before moving"))
    ]
    model = OpenRouterLanguageModel(
        api_key="test-key",
        model="test/provider-model",
        client=cast(OpenRouter, client),
    )

    result = model.complete("What next?", system_prompt="Return safe advice")

    assert result == "observe before moving"
    client.chat.send.assert_called_once_with(
        model="test/provider-model",
        messages=[
            components.ChatSystemMessage(role="system", content="Return safe advice"),
            components.ChatUserMessage(role="user", content="What next?"),
        ],
        stream=False,
    )


def test_missing_api_key_is_rejected() -> None:
    with pytest.raises(ValueError, match="API key"):
        OpenRouterLanguageModel(api_key="")


def test_completion_attaches_local_images_as_data_urls(tmp_path: Path) -> None:
    image = tmp_path / "work" / "00000000.png"
    image.parent.mkdir()
    image.write_bytes(MIN_PNG)
    client = MagicMock()
    client.chat.send.return_value.choices = [
        MagicMock(message=MagicMock(content='sim.run_skill("settle")'))
    ]
    model = OpenRouterLanguageModel(
        api_key="test-key",
        model="test/provider-model",
        client=cast(OpenRouter, client),
    )

    result = model.complete("Which skill?", images=[image])

    assert result == 'sim.run_skill("settle")'
    message = client.chat.send.call_args.kwargs["messages"][0]
    assert message.role == "user"
    assert isinstance(message.content, list)
    assert message.content[0].type == "text"
    assert message.content[0].text == "Which skill?"
    assert message.content[1].type == "image_url"
    assert message.content[1].image_url.url.startswith("data:image/png;base64,")
    assert message.content[2].text == "(image: work/00000000.png)"


def test_completion_rejects_non_image_attachments(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("not an image\n", encoding="utf-8")
    model = OpenRouterLanguageModel(api_key="test-key", model="test/model", client=MagicMock())
    with pytest.raises(ValueError, match="unsupported image type"):
        model.complete("Which skill?", images=[path])
