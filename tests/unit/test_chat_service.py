from unittest.mock import patch

from chatbot_rag.chat.service import ChatService


def test_chat_service_can_be_created() -> None:
    with patch("chatbot_rag.chat.service.ChatOpenAI") as mock_chat_openai:
        service = ChatService()

        mock_chat_openai.assert_called_once_with(
            model="gpt-4o-mini",
            temperature=0,
        )

        assert service.model is mock_chat_openai.return_value
