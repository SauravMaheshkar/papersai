import os
import time
from typing import Dict, List, Optional, TypeAlias

import gradio as gr
import torch
import weave
from transformers import pipeline

from papersai.utils import load_paper_as_context


os.environ["TOKENIZERS_PARALLELISM"] = "false"

HistoryType: TypeAlias = List[Dict[str, str]]

# Initialize the LLM and Weave client
client = weave.init("papersai")
checkpoint: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
pipe = pipeline(
    model=checkpoint,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)


class ChatState:
    """Utility class to store context and last response"""

    def __init__(self):
        self.context = None
        self.last_response = None


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
    input_text = pipe.tokenizer.apply_chat_template(
        history,
        tokenize=False,
    )
    response = pipe(input_text, do_sample=True, top_p=0.95, max_new_tokens=100)[0][
        "generated_text"
    ]
    response = response.split("\nassistant\n")[-1]
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
                text = load_paper_as_context(file_path=file_path)
                doc_context = [x.get_content() for x in text]
                state.context = " ".join(doc_context)[
                    : pipe.model.config.max_position_embeddings
                ]
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
        for character in response:
            history[-1]["content"] += character
            time.sleep(0.02)
            yield history

    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        yield history


def create_interface():
    with gr.Blocks() as demo:
        global state
        state = ChatState()
        gr.Markdown(
            """
            <a href="https://github.com/SauravMaheshkar/papersai">
                <div align="center"><h1>papers.ai</h1></div>
            </a>
            """,
        )
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
