from pathlib import Path

from langchain_core.documents import Document


def load_text_file(path: Path) -> list[Document]:
    content = path.read_text(encoding="utf-8")

    return [
        Document(
            page_content=content,
            metadata={
                "source": str(path),
                "file_type": path.suffix.lstrip("."),
            },
        )
    ]