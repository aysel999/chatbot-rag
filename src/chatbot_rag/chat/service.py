from langchain_ollama import ChatOllama


class ChatService:
    def __init__(self) -> None:
        self.model = ChatOllama(
            model="gemma4:e2b",
            temperature=0,
        )

    def ask(self, message: str) -> str:
        response = self.model.invoke(message)
        return str(response.content)