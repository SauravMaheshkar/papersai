import os
import sys

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from papersai.cli.utils import init_parser
from papersai.engine import Chat
from papersai.utils import load_paper_as_context


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def qna_cli() -> None:
    # initialize parser and console
    parser = init_parser()
    args = parser.parse_args()
    console = Console()

    # Load paper as context
    if args.path_to_pdf is not None:
        context = load_paper_as_context(file_path=args.path_to_pdf)
    else:
        if args.paper_id is None:
            args.paper_id = Prompt.ask("Enter the paper id")
        context = load_paper_as_context(paper_id=args.paper_id)

    chat_agent = Chat(model=args.model, context=context)

    while True:
        try:
            # Get user query
            query = Prompt.ask("Question:")

            # Exit if query is "exit"
            if query.lower().strip() == "exit":
                console.print("\nExiting...")
                sys.exit(0)

            # Get response from chat agent
            response = chat_agent.chat(query=query)

            # Print response
            console.print(
                Panel(
                    response, border_style="blue", title="Response", title_align="left"
                )
            )

        # Exit on KeyboardInterrupt
        except KeyboardInterrupt:
            console.print("\nExiting...")
            sys.exit(0)


if __name__ == "__main__":
    qna_cli()
