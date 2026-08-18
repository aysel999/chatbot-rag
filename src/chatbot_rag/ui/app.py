import streamlit as st
from langchain_ollama import ChatOllama

from chatbot_rag.rag.service import RAGService


st.title("Chatbot RAG")

model = ChatOllama(model="gemma4:e2b")

message = st.chat_input("Ask something...")

if message:
    st.chat_message("user").write(message)

   