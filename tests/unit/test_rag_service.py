from unittest.mock import MagicMock

from langchain_core.documents import Document

from chatbot_rag.rag.service import RAGService


def test_rag_service_answers_using_retrieved_documents() -> None:
    vector_store = MagicMock()
    vector_store.similarity_search.return_value = [
        Document(page_content="Python was created by Guido van Rossum.")
    ]

    fake_model = MagicMock()
    fake_response = MagicMock()
    fake_response.content = "Python was created by Guido van Rossum."
    fake_model.invoke.return_value = fake_response

    service = RAGService(
        vector_store=vector_store,
        model=fake_model,
    )

    result = service.ask("Who created Python?")

    assert result == "Python was created by Guido van Rossum."

    vector_store.similarity_search.assert_called_once_with(
        "Who created Python?",
        k=4,
    )

    fake_model.invoke.assert_called_once()