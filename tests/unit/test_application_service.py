from pathlib import Path
from unittest.mock import MagicMock, patch

from chatbot_rag.application.service import ApplicationService


def test_process_file_builds_vector_store() -> None:
    fake_documents = [MagicMock()]
    fake_chunks = [MagicMock()]
    fake_vector_store = MagicMock()

    with (
        patch(
            "chatbot_rag.application.service.load_documents",
            return_value=fake_documents,
        ) as mock_load,
        patch(
            "chatbot_rag.application.service.split_documents",
            return_value=fake_chunks,
        ) as mock_split,
        patch(
            "chatbot_rag.application.service.create_vector_store",
            return_value=fake_vector_store,
        ) as mock_vector_store,
    ):
        service = ApplicationService()
        result = service.process_file(Path("document.pdf"))

    assert result is fake_vector_store

    mock_load.assert_called_once_with(Path("document.pdf"))
    mock_split.assert_called_once_with(fake_documents)
    mock_vector_store.assert_called_once_with(fake_chunks)