from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


def embed_documents(documents: list[Document]) -> list[list[float]]:
    embeddings = OpenAIEmbeddings()

    return embeddings.embed_documents(
        [document.page_content for document in documents]
    )
