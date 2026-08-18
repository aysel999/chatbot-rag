from pathlib import Path
from unittest.mock import MagicMock, patch

from chatbot_rag.ui.app import create_rag_service, process_uploaded_file


def test_create_rag_service_uses_vector_store_and_gemma() -> None:
    fake_vector_store = MagicMock()
    fake_model = MagicMock()

    with patch(
        "chatbot_rag.ui.app.ChatOllama",
        return_value=fake_model,
    ):
        service = create_rag_service(fake_vector_store)

    assert service.vector_store is fake_vector_store
    assert service.model is fake_model


def test_process_uploaded_file_builds_vector_store() -> None:
    fake_uploaded_file = MagicMock()
    fake_uploaded_file.name = "document.txt"
    fake_uploaded_file.getvalue.return_value = b"test document"
    
    fake_vector_store = MagicMock()

    with (
        patch(
            "chatbot_rag.ui.app.ApplicationService"
        ) as mock_application_service,
    ):
        mock_service = mock_application_service.return_value
        mock_service.process_file.return_value = fake_vector_store

        result = process_uploaded_file(fake_uploaded_file)

    assert result is fake_vector_store
    mock_service.process_file.assert_called_once()