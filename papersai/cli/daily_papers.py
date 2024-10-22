from datetime import datetime, timedelta
from typing import Dict, List

import requests
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.prompt import Confirm, Prompt

from papersai.cli.utils import init_model, init_parser
from papersai.engine.summarize import get_summary
from papersai.utils import load_paper_as_context


DAILY_PAPERS_ENDPOINT: str = "https://huggingface.co/api/daily_papers"


def info_to_md_list(id_title_dict_list: List[Dict[str, str]]) -> str:
    txt = ""
    for index, item in enumerate(id_title_dict_list, start=1):
        txt += f"{index}. {item['id']} | {item['name']}\n"
    return txt


def get_date() -> str:
    now = datetime.now()

    if now.hour >= 9:
        date_to_check = now
    else:
        date_to_check = now - timedelta(days=1)

    # Adjust for weekends
    if date_to_check.weekday() == 5:
        date_to_check -= timedelta(days=1)
    elif date_to_check.weekday() == 6:
        date_to_check -= timedelta(days=2)

    return date_to_check.strftime("%Y-%m-%d")


def fetch_ids() -> List[str]:
    ids = []
    params: Dict[str, str] = {"date": get_date()}
    response = requests.get(url=DAILY_PAPERS_ENDPOINT, params=params)

    # collect paper ids
    for element in response.json():
        _id = element["paper"]["id"]
        ids.append(_id)

    return ids


def fetch_papers() -> List[Dict[str, str]]:
    info = []
    params: Dict[str, str] = {"date": get_date()}
    response = requests.get(url=DAILY_PAPERS_ENDPOINT, params=params)

    for element in response.json():
        _id = element["paper"]["id"]
        name = element["paper"]["title"]
        info.append({"id": _id, "name": name})

    return info


def daily_papers_cli():
    # Define Argument Parser
    parser = init_parser()
    args = parser.parse_args()

    # Initialize Model
    llm = init_model(args.model)
    Settings.llm = llm

    Settings.embed_model = HuggingFaceEmbedding(
        model_name=args.embedding_model, trust_remote_code=True
    )

    console = Console()
    info = fetch_papers()
    ids = fetch_ids()

    console.print(Panel(info_to_md_list(info), title=get_date()))

    if Confirm.ask("Do you want the summary of any paper in particular?", default=True):
        paper_id = Prompt.ask("Enter the paper id", choices=ids)
        with Progress(transient=True) as progress:
            # Download and load paper
            load_docs_task = progress.add_task(
                "[cyan]downloading and loading the paper as context for LLM", total=1
            )
            context = load_paper_as_context(paper_id=paper_id)
            progress.update(load_docs_task, completed=True)

            # Generate Summaries
            generate_summaries_task = progress.add_task(
                "[cyan]generating summary", total=len(context)
            )
            summary = get_summary(
                context, rich_metadata=[progress, generate_summaries_task]
            )
            progress.console.print(Panel(summary, title=f"Summary for {paper_id}"))
        if Confirm.ask(
            "Do you want to ask any questions about this paper?", default=True
        ):
            index = VectorStoreIndex.from_documents(context)
            chat_engine = index.as_chat_engine()
            chat_engine.chat_repl()


if __name__ == "__main__":
    daily_papers_cli()
