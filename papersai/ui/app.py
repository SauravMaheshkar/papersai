import time
from typing import Dict, List, Optional, TypeAlias

import gradio as gr
import weave
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages.base import BaseMessage

from papersai.utils import load_paper_as_context


HistoryType: TypeAlias = List[Dict[str, str]]

load_dotenv()

# Initialize the LLM and Weave client
client = weave.init("papersai")
llm = init_chat_model(
    model="claude-3-haiku-20240307",
    model_provider="anthropic",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
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
    if x.liked:
        call.feedback.add_reaction("👍")
    else:
        call.feedback.add_reaction("👎")


@weave.op()
def invoke(messages: List[BaseMessage]) -> BaseMessage:
    """
    Simple wrapper around the llm.invoke method wrapped in a weave op

    Args:
        messages (List[BaseMessage]): List of messages to pass to the model

    Returns:
        BaseMessage: Response from the model
    """
    return llm.invoke(messages)


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
            except Exception as e:
                history.append(
                    {"role": "assistant", "content": f"Error loading file: {str(e)}"}
                )

    # Handle text input
    if isinstance(message, dict) and message.get("text"):
        history.append({"role": "user", "content": message["text"]})

    return history, gr.MultimodalTextbox(value=None, interactive=True)


def format_history(history: HistoryType) -> List[BaseMessage]:
    """
    Convert chat history to LangChain message format

    Args:
        history (HistoryType): Chat history

    Returns:
        List[BaseMessage]: List of messages in LangChain format
    """
    messages = []

    # Add context as the first message if available
    if state.context is not None:
        messages.append(HumanMessage(content=f"Context: {state.context}"))

    # Add the rest of the conversation
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    return messages


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

    # Convert history to LangChain format
    messages = format_history(history)

    try:
        # Get response from LLM
        response, call = invoke.call(messages)
        state.last_response = call

        # Add empty assistant message
        history.append({"role": "assistant", "content": ""})

        # Stream the response
        for character in response.content:
            history[-1]["content"] += character
            time.sleep(0.02)
            yield history

    except Exception as e:
        history.append({"role": "assistant", "content": f"Error: {str(e)}"})
        yield history


def create_interface():
    with gr.Blocks() as demo:
        with weave.attributes({"session": "local"}):
            global state
            state = ChatState()
            gr.Markdown(
                """
                <a href="https://github.com/SauravMaheshkar/papersai">
                    <div align="center">
                        <h1>papers.ai</h1>
                    </div>
                </a>""",
            )
            chatbot = gr.Chatbot(
                show_label=False,
                height=600,
                type="messages",
                show_copy_all_button=True,
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
