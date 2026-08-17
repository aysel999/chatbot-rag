import streamlit as st

from chatbot_rag.chat.service import ChatService


st.title("Chatbot RAG")

service = ChatService()

message = st.chat_input("Ask something...")

if message:
    st.chat_message("user").write(message)

    response = service.ask(message)

    st.chat_message("assistant").write(response)