from pathlib import Path
from unittest.mock import Mock, patch

from chatbot_rag.ingestion.pdf import load_pdf_file


def test_load_pdf_file_returns_one_document_for_each_page(tmp_path: Path) -> None:
    file_path = tmp_path / "guide.pdf"

    first_page = Mock()
    first_page.extract_text.return_value = "First PDF page"

    second_page = Mock()
    second_page.extract_text.return_value = "Second PDF page"

    with patch("chatbot_rag.ingestion.pdf.PdfReader") as mock_pdf_reader:
        mock_pdf_reader.return_value.pages = [first_page, second_page]

        documents = load_pdf_file(file_path)

    mock_pdf_reader.assert_called_once_with(file_path)

    assert [document.page_content for document in documents] == [
        "First PDF page",
        "Second PDF page",
    ]
    assert [document.metadata for document in documents] == [
        {
            "source": str(file_path),
            "file_type": "pdf",
            "page": 1,
        },
        {
            "source": str(file_path),
            "file_type": "pdf",
            "page": 2,
        },
    ]