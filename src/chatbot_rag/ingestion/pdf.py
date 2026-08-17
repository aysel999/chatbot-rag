from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader


def load_pdf_file(path: Path) -> list[Document]:
    reader = PdfReader(path)

    return [
        Document(
            page_content=page.extract_text() or "",
            metadata={
                "source": str(path),
                "file_type": "pdf",
                "page": page_number,
            },
        )
        for page_number, page in enumerate(reader.pages, start=1)
    ]