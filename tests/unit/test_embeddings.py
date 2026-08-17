from unittest.mock import MagicMock, patch

from langchain_core.documents import Document

from chatbot_rag.retrieval.embeddings import embed_documents


def test_embed_documents_returns_embeddings() -> None:
    documents = [
        Document(page_content="Python is a programming language."),
        Document(page_content="Paris is the capital of France."),
    ]

    fake_embeddings = MagicMock()
    fake_embeddings.embed_documents.return_value = [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    with patch(
        "chatbot_rag.retrieval.embeddings.OllamaEmbeddings",
        return_value=fake_embeddings,
    ) as mock_embeddings:
        result = embed_documents(documents)

    assert result == [
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
    ]

    mock_embeddings.assert_called_once_with(
        model="nomic-embed-text",
    )

    fake_embeddings.embed_documents.assert_called_once_with(
        [
            "Python is a programming language.",
            "Paris is the capital of France.",
        ]
    )