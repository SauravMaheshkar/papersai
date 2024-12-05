import weave
from litellm import completion


__all__ = ["Chat"]


CHAT_SYSTEM_PROMPT: str = (
    "You are an expert in the field of AI and Machine Learning."
    "You are chatting with a researcher who is conducting research "
    "based on the following paper. You are expected to provide "
    "insightful responses to the researcher's queries.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    "Researcher:"
)


class Chat(weave.Model):
    model: str
    context: str
    history: list[dict[str, str]]

    def __init__(self, context, model) -> None:
        super().__init__(model=model, context=context, history=[])
        self.history = [
            {
                "role": "system",
                "content": CHAT_SYSTEM_PROMPT.format(context_str=context),
            }
        ]

    @weave.op()
    def chat(self, query: str) -> str:
        # Add user query to history
        self.history.append({"role": "user", "content": query})

        # Get response from LLM
        response = completion(
            model=self.model,
            messages=self.history,
        )
        response = str(response.choices[0].message.content)

        # Add assistant response to history
        self.history.append({"role": "assistant", "content": response})

        return response
