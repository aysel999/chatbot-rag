from typing import Any


class RAGService:
    def __init__(self, vector_store: Any, model: Any) -> None:
        self.vector_store = vector_store
        self.model = model

    def ask(self, question: str) -> str:
        documents = self.vector_store.similarity_search(
            question,
            k=4,
        )

        context = "\n\n".join(
            document.page_content for document in documents
        )

        prompt = f"""Answer the question using only the provided context.

If the answer is not contained in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""

        response = self.model.invoke(prompt)

        return response.content