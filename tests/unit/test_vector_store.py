from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from chatbot_rag.retrieval.vector_store import create_vector_store


def test_create_vector_store_adds_documents() -> None:
    documents = [
        Document(
            page_content="Python is a programming language.",
            metadata={"source": "python.txt"},
        ),
        Document(
            page_content="Paris is the capital of France.",
            metadata={"source": "france.txt"},
        ),
    ]

    fake_embeddings = MagicMock()
    fake_vector_store = MagicMock()

    with (
        patch(
            "chatbot_rag.retrieval.vector_store.OllamaEmbeddings",
            return_value=fake_embeddings,
        ),
        patch(
            "chatbot_rag.retrieval.vector_store.Chroma.from_documents",
            return_value=fake_vector_store,
        ) as mock_from_documents,
    ):
        result = create_vector_store(documents)

    assert result is fake_vector_store

    mock_from_documents.assert_called_once_with(
        documents=documents,
        embedding=fake_embeddings,
    )