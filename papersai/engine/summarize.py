import weave
from litellm import completion


__all__ = ["Summarizer"]


# References:
# * https://github.com/run-llama/llama_index/blob/eeb068f75ce2588636f685c5fb91ee88760ea5ce/llama-index-core/llama_index/core/prompts/default_prompts.py#L10
SUMMARY_PROMPT: str = (
    "Write a summary of the following. Try to use only the "
    "information provided. "
    "Try to include as many key details as possible.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    'SUMMARY:"""\n'
)


class Summarizer(weave.Model):
    model: str

    def __init__(self, model) -> None:
        super().__init__(model=model)

    @weave.op()
    def summarize(
        self,
        context: str,
        rich_metadata=None,
    ) -> str:
        response = completion(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": SUMMARY_PROMPT.format(context_str=context),
                }
            ],
        )

        if rich_metadata:
            rich_metadata[0].update(rich_metadata[1], advance=1)

        return str(response.choices[0].message.content)
