from collections.abc import Callable
from pathlib import Path

from langchain_core.documents import Document
from chatbot_rag.ingestion.docx import load_docx_file
from chatbot_rag.ingestion.pdf import load_pdf_file
from chatbot_rag.ingestion.text import load_text_file


LOADERS: dict[str, Callable[[Path], list[Document]]] = {
    ".txt": load_text_file,
    ".md": load_text_file,
    ".pdf": load_pdf_file,
    ".docx": load_docx_file,
}


def load_documents(path: Path) -> list[Document]:
    file_type = path.suffix.lower()
    loader = LOADERS.get(file_type)

    if loader is None:
        raise ValueError(f"Unsupported file type: {file_type}")

    return loader(path)