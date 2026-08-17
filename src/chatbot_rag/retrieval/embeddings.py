from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings


def embed_documents(documents: list[Document]) -> list[list[float]]:
    embeddings = OllamaEmbeddings(
        model="nomic-embed-text",
    )

    return embeddings.embed_documents(
        [document.page_content for document in documents]
    )