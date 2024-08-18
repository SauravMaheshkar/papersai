from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from papersai.cli.utils import init_model, init_parser
from papersai.engine.summarize import get_summary
from papersai.utils import load_paper_as_context


def summarize_cli():
    # Define Argument Parser
    parser = init_parser()
    args = parser.parse_args()

    # Initialize Model
    llm = init_model(args.model)
    Settings.llm = llm

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
