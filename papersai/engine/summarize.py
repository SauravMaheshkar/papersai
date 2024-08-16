from typing import List

from llama_index.core import Document
from llama_index.core.response_synthesizers import TreeSummarize


def get_summary(context: List[Document], verbose: bool = False) -> str:
    summarizer = TreeSummarize(verbose=verbose)
    summary = summarizer.get_response(
        "Summarize the paper", [doc.text for doc in context]
    )

    return summary  # type: ignore
