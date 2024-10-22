import os
from typing import List, Optional

import requests
from llama_index.core import Document, SimpleDirectoryReader


ARXIV_PDF_URL: str = "https://arxiv.org/pdf/"


def download_paper(paper_id: str, path: Optional[str] = "artifacts") -> None:
    """
    Download a function from arxiv given the paper id.

    Args:
        paper_id (str): The id of the paper to download.
        path (Optional[str], optional): Defaults to "artifacts"

    Raises:
        ValueError: If paper id is not valid.
        AssertionError: If paper id is not provided.

    Returns:
        None
    """
    os.makedirs(path, exist_ok=True)  # type: ignore[arg-type]

    assert paper_id is not None, "You must provide a ID"

    _url = ARXIV_PDF_URL + paper_id
    response = requests.get(_url)
    if response.status_code == 200:
        _url = ARXIV_PDF_URL + paper_id
        response = requests.get(_url)
        with open(f"{path}/{paper_id}.pdf", "wb") as f:
            f.write(response.content)
    else:
        raise ValueError(
            f"download for paper id {paper_id} failed with error code {response.status_code}"  # noqa: E501
        )


def load_paper_as_context(
    paper_id: str, path: Optional[str] = "artifacts", verbose: Optional[bool] = False
) -> List[Document]:
    """
    Downloads a paper from arxiv given the paper id and loads it as context.

    Args:
        paper_id (str): The id of the paper to download.
        verbose (bool, optional): Defaults to False.
        path (Optional[str], optional): Defaults to "artifacts"

    Raises:
        ValueError: If paper id is not valid.
        AssertionError: If paper id is not provided.

    Returns:
        (List[Document]): list of llamaindex document objects for the paper.
    """
    download_paper(paper_id=paper_id)
    reader = SimpleDirectoryReader(
        input_dir=f"{path}/",
        input_files=[f"{path}/{paper_id}.pdf"],
    )
    context = reader.load_data(show_progress=verbose)  # type: ignore[arg-type]

    return context
