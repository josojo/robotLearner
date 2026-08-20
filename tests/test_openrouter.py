from typing import cast
from unittest.mock import MagicMock

import pytest
from openrouter import OpenRouter, components

from robot_learner.openrouter import OpenRouterLanguageModel


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
