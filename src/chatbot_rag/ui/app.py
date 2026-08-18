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
    temp_path = Path("uploaded_document" + suffix)

    temp_path.write_bytes(uploaded_file.getvalue())

    service = ApplicationService()
    return service.process_file(temp_path)


st.title("Chatbot RAG")

message = st.chat_input("Ask something...")

if message:
    st.chat_message("user").write(message)