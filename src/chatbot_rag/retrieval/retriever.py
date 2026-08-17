from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore


def retrieve_documents(
    vector_store: VectorStore,
    query: str,
    k: int = 4,
) -> list[Document]:
    return vector_store.similarity_search(query, k=k)