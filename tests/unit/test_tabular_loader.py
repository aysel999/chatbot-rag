from pathlib import Path

from chatbot_rag.ingestion.tabular import load_csv_file


def test_load_csv_file_returns_one_document_for_each_row(tmp_path: Path) -> None:
    file_path = tmp_path / "people.csv"
    file_path.write_text(
        "name,role\nAysel,Developer\nLeyla,Designer\n",
        encoding="utf-8",
    )

    documents = load_csv_file(file_path)

    assert [document.page_content for document in documents] == [
        "name: Aysel\nrole: Developer",
        "name: Leyla\nrole: Designer",
    ]
    assert [document.metadata for document in documents] == [
        {
            "source": str(file_path),
            "file_type": "csv",
            "row": 1,
        },
        {
            "source": str(file_path),
            "file_type": "csv",
            "row": 2,
        },
    ]
