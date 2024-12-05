from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from papersai.cli.utils import init_parser
from papersai.engine import Summarizer
from papersai.utils import load_paper_as_context


def summarize_cli() -> None:
    # Define Argument Parser
    parser = init_parser()
    args = parser.parse_args()

    # Load paper as context
    if args.path_to_pdf is not None:
        context = load_paper_as_context(file_path=args.path_to_pdf)
    else:
        if args.paper_id is None:
            args.paper_id = Prompt.ask("Enter the paper id")
        context = load_paper_as_context(paper_id=args.paper_id)

    # Generate summary
    summary = Summarizer(model=args.model).summarize(context=context)

    # Display summary
    console = Console()
    console.print(Panel(summary, title="Summary"))


if __name__ == "__main__":
    summarize_cli()
