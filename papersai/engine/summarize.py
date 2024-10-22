from typing import List

from llama_index.core import Document
from llama_index.core.response_synthesizers import TreeSummarize


def get_summary(
    context: List[Document], verbose: bool = False, rich_metadata=None
) -> str:
    summarizer = TreeSummarize(verbose=verbose)
    summary = summarizer.get_response(
        "Summarize the paper", [doc.text for doc in context]
    )

    if rich_metadata:
        rich_metadata[0].update(rich_metadata[1], advance=1)

    return str(summary)
