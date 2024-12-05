import os
import time
from typing import Dict, List, Optional, TypeAlias

import gradio as gr
import weave
from litellm import completion

from papersai import Summarizer, load_paper_as_context


os.environ["TOKENIZERS_PARALLELISM"] = "false"

HistoryType: TypeAlias = List[Dict[str, str]]

# Initialize the Weave client
client = weave.init("papersai")


class ChatState:
    """Utility class to store context and last response"""

    def __init__(self):
        self.context = None
        self.last_response = None


def summarize_paper(paper_id: str) -> str:
    """
    Summarize the paper with the given arxiv ID.

    Args:
        paper_id (str): arxiv paper ID

    Returns:
        str: summarized text from the paper
    """
    gr.Info("Downloading paper...")
    context = load_paper_as_context(paper_id=paper_id)
    gr.Info("Summarizing ...")
    summary = Summarizer(model="claude-3-5-sonnet-20240620").summarize(context=context)
    return gr.Textbox(value=summary, visible=True)


def record_feedback(x: gr.LikeData) -> None:
    """
    Logs user feedback on the assistant's response in the form of a
    like/dislike reaction.

    Reference:
    * https://weave-docs.wandb.ai/guides/tracking/feedback

    Args:
        x (gr.LikeData): User feedback data

    Returns:
        None
    """
    call = state.last_response

    # Remove any existing feedback before adding new feedback
    for existing_feedback in list(call.feedback):
        call.feedback.purge(existing_feedback.id)

    if x.liked:
        call.feedback.add_reaction("👍")
    else:
        call.feedback.add_reaction("👎")


@weave.op()
def invoke(history: HistoryType):
    """
    Simple wrapper around llm inference wrapped in a weave op

    Args:
        history (HistoryType): Chat history

    Returns:
        BaseMessage: Response from the model
    """
    response = completion(model="claude-3-5-sonnet-20240620", messages=history)
    return response


def update_state(history: HistoryType, message: Optional[Dict[str, str]]):
    """
    Update history and app state with the latest user input.

    Args:
        history (HistoryType): Chat history
        message (Optional[Dict[str, str]]): User input message

    Returns:
        Tuple[HistoryType, gr.MultimodalTextbox]: Updated history and chat input
    """
    if message is None:
        return history, gr.MultimodalTextbox(value=None, interactive=True)

    # Initialize history if None
    if history is None:
        history = []

    # Handle file uploads without adding to visible history
    if isinstance(message, dict) and "files" in message:
        for file_path in message["files"]:
            try:
                state.context = load_paper_as_context(file_path=file_path)
                history.append(
                    {"role": "system", "content": f"Context: {state.context}\n"}
                )
            except Exception as e:
                history.append(
                    {"role": "assistant", "content": f"Error loading file: {str(e)}"}
                )

    # Handle text input
    if isinstance(message, dict) and message.get("text"):
        history.append({"role": "user", "content": message["text"]})

    return history, gr.MultimodalTextbox(value=None, interactive=True)


def bot(history: HistoryType):
    """
    Generate response from the LLM and stream it back to the user.

    Args:
        history (HistoryType): Chat history

    Yields:
        response from the LLM
    """
    if not history:
        return history

    try:
        # Get response from LLM
        response, call = invoke.call(history)
        state.last_response = call

        # Add empty assistant message
        history.append({"role": "assistant", "content": ""})

        # Stream the response
        for character in response.choices[0].message.content:
            history[-1]["content"] += character
            time.sleep(0.02)
            yield history

    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        yield history


def create_interface():
    with gr.Blocks() as demo:
        gr.Markdown(
            """
                <a href="https://github.com/SauravMaheshkar/papersai">
                    <div align="center"><h1>papers.ai</h1></div>
                </a>
                """,
        )

        with gr.Tab("Summary"):
            input_id = gr.Textbox(
                label="Paper ID",
                placeholder="Enter the arxiv paper ID",
            )
            output_summary = gr.Textbox(
                label="Summary",
                placeholder="Summary will be displayed here",
                visible=False,
            )
            btn = gr.Button(
                value="Summarize",
                variant="primary",
            )
            btn.click(
                fn=summarize_paper,
                inputs=[input_id],
                outputs=[output_summary],
            )

        with gr.Tab("Chat"):
            global state
            state = ChatState()
            chatbot = gr.Chatbot(
                show_label=False,
                height=600,
                type="messages",
                show_copy_all_button=True,
                placeholder="Upload a research paper and ask questions!!",
            )

            chat_input = gr.MultimodalTextbox(
                interactive=True,
                file_count="single",
                placeholder="Upload a document or type your message...",
                show_label=False,
                stop_btn=True,
            )

            chat_msg = chat_input.submit(
                fn=update_state,
                inputs=[chatbot, chat_input],
                outputs=[chatbot, chat_input],
            )

            bot_msg = chat_msg.then(  # noqa: F841
                fn=bot, inputs=[chatbot], outputs=chatbot, api_name="bot_response"
            )

            chatbot.like(
                fn=record_feedback,
                inputs=None,
                outputs=None,
                like_user_message=True,
            )

    return demo


def main():
    demo = create_interface()
    demo.launch(share=False)


if __name__ == "__main__":
    main()
