import argparse

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from papersai.engine.summarize import get_summary
from papersai.utils import load_paper_as_context


def summarize_cli():
    # Define Argument Parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Which model to use, one of openai, anthropic or some model from the huggingface hub",  # noqa: E501
        default="anthropic",
    )
    parser.add_argument(
        "--embedding_model",
        help="which embedding model to use",
        default="BAAI/bge-small-en-v1.5",
    )
    parser.add_argument(
        "--paper_id",
        help="arxiv id of the paper you want to summarize",
        default=None,
    )
    args = parser.parse_args()

    # Initialize Model
    if args.model == "anthropic":
        load_dotenv()
        from llama_index.llms.anthropic import Anthropic

        Settings.llm = Anthropic(temperature=0.0, model="claude-3-haiku-20240307")
    else:
        raise NotImplementedError("Only supports Anthropic as of now")

    # Initialize Embedding Model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=args.embedding_model, trust_remote_code=True
    )

    # Determine paper_id
    paper_id = args.paper_id
    if paper_id is None:
        paper_id = Prompt.ask("Enter the paper id")

    # Get context and create summary
    context = load_paper_as_context(paper_id=paper_id)
    summary = get_summary(context)

    # Display summary
    console = Console()
    console.print(Panel(summary, title=f"Summary for {paper_id}"))


if __name__ == "__main__":
    summarize_cli()
