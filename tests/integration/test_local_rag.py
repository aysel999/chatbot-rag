from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def test_local_chroma_can_store_and_retrieve_documents() -> None:
    documents = [
        Document(
            page_content="Python was created by Guido van Rossum.",
            metadata={"source": "python.txt"},
        ),
        Document(
            page_content="Paris is the capital of France.",
            metadata={"source": "france.txt"},
        ),
    ]

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
    )

    results = vector_store.similarity_search(
        "Who created Python?",
        k=1,
    )

    assert len(results) == 1
    assert "Guido van Rossum" in results[0].page_content