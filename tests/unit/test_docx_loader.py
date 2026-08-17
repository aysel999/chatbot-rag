from pathlib import Path
from unittest.mock import Mock, patch

from chatbot_rag.ingestion.docx import load_docx_file


def test_load_docx_file_returns_paragraph_text_and_metadata(
    tmp_path: Path,
) -> None:
    file_path = tmp_path / "guide.docx"

    first_paragraph = Mock()
    first_paragraph.text = "First paragraph"

    second_paragraph = Mock()
    second_paragraph.text = "Second paragraph"

    with patch("chatbot_rag.ingestion.docx.DocxDocument") as mock_document:
        mock_document.return_value.paragraphs = [
            first_paragraph,
            second_paragraph,
        ]

        documents = load_docx_file(file_path)

    mock_document.assert_called_once_with(file_path)

    assert len(documents) == 1
    assert documents[0].page_content == "First paragraph\nSecond paragraph"
    assert documents[0].metadata == {
        "source": str(file_path),
        "file_type": "docx",
    }