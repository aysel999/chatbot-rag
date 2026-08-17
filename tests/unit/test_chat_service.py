from unittest.mock import patch

from chatbot_rag.chat.service import ChatService


def test_chat_service_can_be_created() -> None:
    with patch("chatbot_rag.chat.service.ChatOllama") as mock_chat_ollama:
        service = ChatService()

    mock_chat_ollama.assert_called_once_with(
        model="gemma4:e2b",
        temperature=0,
    )
    assert service.model is mock_chat_ollama.return_value