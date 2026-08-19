import tempfile
from pathlib import Path

import streamlit as st
from langchain_ollama import ChatOllama

from chatbot_rag.application.service import ApplicationService
from chatbot_rag.rag.service import RAGService


def create_rag_service(vector_store) -> RAGService:
    model = ChatOllama(model="gemma4:e2b")

    return RAGService(
        vector_store=vector_store,
        model=model,
    )


def process_uploaded_file(uploaded_file):
    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        delete=False,
    ) as temp_file:
        temp_file.write(uploaded_file.getvalue())
        temp_path = Path(temp_file.name)

    try:
        service = ApplicationService()
        return service.process_file(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)


st.title("Chatbot RAG")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["txt", "md", "pdf", "docx"],
)

if uploaded_file:
    if st.button("Process document"):
        with st.spinner("Processing document..."):
            st.session_state.vector_store = process_uploaded_file(
                uploaded_file
            )

        st.success("Document processed successfully!")

if "vector_store" in st.session_state:
    service = create_rag_service(st.session_state.vector_store)

    message = st.chat_input("Ask something about your document...")

    if message:
        st.chat_message("user").write(message)

        with st.spinner("Thinking..."):
            response = service.ask(message)

        st.chat_message("assistant").write(response)