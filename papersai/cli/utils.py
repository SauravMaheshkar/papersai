import argparse

from llama_index.core.base.llms.base import BaseLLM


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Which model to use, one of anthropic or some model from the huggingface hub",  # noqa: E501
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

    return parser


def init_model(model_id: str) -> BaseLLM:
    """
    Utility Function to initialize a LLM

    Args:
        model_id (str): Which model to use, one of anthropic or
            some model from the huggingface hub

    Returns:
        A LLM of type `BaseLLM` from llamaindex

    Raises:
        NotImplementedError
    """
    if model_id == "anthropic":
        from dotenv import load_dotenv
        from llama_index.llms.anthropic import Anthropic

        # Load Credentials
        load_dotenv()

        return Anthropic(temperature=0.0, model="claude-3-haiku-20240307")
    else:
        raise NotImplementedError("Only Anthropic Models are officially supported")
