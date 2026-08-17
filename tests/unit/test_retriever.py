from unittest.mock import MagicMock

from langchain_core.documents import Document

from chatbot_rag.retrieval.retriever import retrieve_documents


def test_retrieve_documents_returns_relevant_documents() -> None:
    documents = [
        Document(page_content="Python is a programming language."),
        Document(page_content="Paris is the capital of France."),
    ]

    fake_vector_store = MagicMock()

    fake_vector_store.similarity_search.return_value = [
        documents[0],
    ]

    result = retrieve_documents(
        vector_store=fake_vector_store,
        query="What is Python?",
    )

    assert result == [documents[0]]

    fake_vector_store.similarity_search.assert_called_once_with(
        "What is Python?",
        k=4,
    )