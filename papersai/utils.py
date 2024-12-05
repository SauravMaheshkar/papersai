import os
from typing import Optional

import pymupdf4llm
import requests
import weave


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
    os.makedirs(path, exist_ok=True)

    assert paper_id is not None, "You must provide a ID"

    if os.path.exists(f"{path}/{paper_id}.pdf"):
        return
    else:
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


@weave.op
def load_paper_as_context(
    paper_id: Optional[str] = None,
    file_path: Optional[str] = None,
    save_path: Optional[str] = "artifacts",
) -> str:
    """
    Given either the arxiv id or the path to a paper, this function
    downloads the paper and loads it as context.

    Args:
        paper_id (str, optional): Defaults to None, the arxiv id of the paper
        file_path (str, optional): Defaults to None, path to the
            paper if it is already downloaded.
        save_path (str, optional): Defaults to "artifacts", path to
            save the paper.
        verbose (bool, optional): Defaults to False.

    Raises:
        ValueError: If paper id is not valid.
        AssertionError: If paper id or file path is not provided

    Returns:
        str: parsed text from the paper
    """
    # If paper_id is provided
    if paper_id is not None:
        download_paper(paper_id=paper_id)
        context = pymupdf4llm.to_markdown(
            f"{save_path}/{paper_id}.pdf", show_progress=False
        )
    # If file_path is provided
    elif file_path is not None:
        context = pymupdf4llm.to_markdown(file_path, show_progress=False)
    else:
        raise AssertionError("Either paper_id or file_path must be provided")

    return context
