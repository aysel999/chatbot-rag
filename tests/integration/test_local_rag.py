from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import ChatOllama, OllamaEmbeddings

from chatbot_rag.rag.service import RAGService


def test_local_rag_can_retrieve_and_answer() -> None:
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

    model = ChatOllama(model="gemma4:e2b")

    service = RAGService(
        vector_store=vector_store,
        model=model,
    )

    result = service.ask("Who created Python?")

    assert "Guido van Rossum" in result