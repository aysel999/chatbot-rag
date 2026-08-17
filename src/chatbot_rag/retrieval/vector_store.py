from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def create_vector_store(documents: list[Document]) -> Chroma:
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )