import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import argparse

from dotenv import load_dotenv
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from rich.prompt import Prompt

from papersai.index import create_index
from papersai.utils import load_paper_as_context


def qna_cli():
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

    # Create Index
    index = create_index(context=context)
    chat_engine = index.as_chat_engine()
    chat_engine.chat_repl()


if __name__ == "__main__":
    qna_cli()
