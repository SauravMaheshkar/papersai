from typing import List, Optional

from llama_index.core import Document, VectorStoreIndex

SUPPORTED_INDICES: List[str] = ["llamaindex_vectorstore"]


def create_index(
    context: List[Document], variant: Optional[str] = "llamaindex_vectorstore"
):
    assert variant in SUPPORTED_INDICES, f"Only {SUPPORTED_INDICES} are supported atm!"

    if variant == "llamaindex_vectorstore":
        return VectorStoreIndex.from_documents(context)
