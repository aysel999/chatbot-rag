from langchain_core.documents import Document

from chatbot_rag.retrieval.chunking import split_documents


def test_split_documents_creates_smaller_chunks() -> None:
    documents = [
        Document(
            page_content="A" * 100,
            metadata={"source": "test.txt"},
        )
    ]

    chunks = split_documents(
        documents,
        chunk_size=40,
        chunk_overlap=10,
    )

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 40 for chunk in chunks)


def test_split_documents_preserves_metadata() -> None:
    documents = [
        Document(
            page_content="This is a test document.",
            metadata={"source": "test.txt", "file_type": "txt"},
        )
    ]

    chunks = split_documents(documents)

    assert chunks[0].metadata == {
        "source": "test.txt",
        "file_type": "txt",
    }