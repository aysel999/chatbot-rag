from pathlib import Path

from chatbot_rag.ingestion.text import load_text_file


def test_load_text_file_returns_content_and_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.txt"
    file_path.write_text("RAG test content", encoding="utf-8")

    documents = load_text_file(file_path)

    assert len(documents) == 1
    assert documents[0].page_content == "RAG test content"
    assert documents[0].metadata == {
        "source": str(file_path),
        "file_type": "txt",
    }


def test_load_markdown_file_returns_content_and_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "notes.md"
    file_path.write_text("# RAG notes", encoding="utf-8")

    documents = load_text_file(file_path)

    assert len(documents) == 1
    assert documents[0].page_content == "# RAG notes"
    assert documents[0].metadata == {
        "source": str(file_path),
        "file_type": "md",
    }