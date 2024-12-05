import argparse


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        help="Which model to use, any one of the litellm supported models can be used",  # noqa: E501
        default="claude-3-5-sonnet-20240620",
    )
    parser.add_argument(
        "--paper_id",
        help="arxiv id of the paper you want to summarize",
        default=None,
    )
    parser.add_argument(
        "--path_to_pdf",
        help="path to the pdf file you want to load",
        default=None,
    )

    return parser
