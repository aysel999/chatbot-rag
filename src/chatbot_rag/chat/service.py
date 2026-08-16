from langchain_openai import ChatOpenAI


class ChatService:
    def __init__(self) -> None:
        self.model = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )

    def ask(self, message: str) -> str:
        response = self.model.invoke(message)
        return response.content
