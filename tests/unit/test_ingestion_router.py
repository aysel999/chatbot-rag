from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from chatbot_rag.ingestion.router import LOADERS, load_documents


def test_load_documents_uses_text_loader_for_text_file(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("RAG test content", encoding="utf-8")

    documents = load_documents(file_path)

    assert documents[0].page_content == "RAG test content"


def test_load_documents_uses_pdf_loader_for_pdf_file(tmp_path: Path) -> None:
    file_path = tmp_path / "guide.pdf"
    mock_load_pdf_file = Mock(return_value=[])

    with patch.dict(LOADERS, {".pdf": mock_load_pdf_file}):
        documents = load_documents(file_path)

    mock_load_pdf_file.assert_called_once_with(file_path)
    assert documents == []


def test_load_documents_rejects_unsupported_file_type(tmp_path: Path) -> None:
    file_path = tmp_path / "data.csv"

    with pytest.raises(ValueError, match=r"Unsupported file type: \.csv"):
        load_documents(file_path)