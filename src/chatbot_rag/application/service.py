from pathlib import Path

from langchain_core.vectorstores import VectorStore

from chatbot_rag.ingestion.router import load_documents
from chatbot_rag.retrieval.chunking import split_documents
from chatbot_rag.retrieval.vector_store import create_vector_store


class ApplicationService:
    def process_file(self, path: Path) -> VectorStore:
        documents = load_documents(path)
        chunks = split_documents(documents)
        return create_vector_store(chunks)