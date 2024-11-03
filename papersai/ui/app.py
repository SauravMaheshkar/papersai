import time

import gradio as gr
import weave
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage

from papersai.utils import load_paper_as_context


load_dotenv()
client = weave.init("papersai")


class ChatState:
    def __init__(self):
        self.context = None
        self.last_response = None


chat_state = ChatState()

# Initialize the LLM
llm = ChatAnthropic(
    model="claude-3-haiku-20240307",
    temperature=0,
    max_tokens=1024,
    timeout=None,
    max_retries=2,
)


@weave.op()
def invoke(messages):
    return llm.invoke(messages)


def print_like_dislike(x: gr.LikeData):
    call = chat_state.last_response
    if x.liked:
        call.feedback.add_reaction("👍")
    else:
        call.feedback.add_reaction("👎")


def add_message(history, message):
    """Convert uploaded files and text into chat history format"""
    if message is None:
        return history, gr.MultimodalTextbox(value=None, interactive=True)

    # Initialize history if None
    if history is None:
        history = []

    # Handle file uploads without adding to visible history
    if isinstance(message, dict) and "files" in message:
        for file_path in message["files"]:
            try:
                chat_state.context = load_paper_as_context(file_path=file_path)
            except Exception as e:
                history.append(
                    {"role": "assistant", "content": f"Error loading file: {str(e)}"}
                )

    # Handle text input
    if isinstance(message, dict) and message.get("text"):
        history.append({"role": "user", "content": message["text"]})

    return history, gr.MultimodalTextbox(value=None, interactive=True)


def format_chat_history(history):
    """Convert chat history to LangChain message format"""
    messages = []

    # Add context as the first message if available
    if chat_state.context is not None:
        messages.append(HumanMessage(content=f"Context: {chat_state.context}"))

    # Add the rest of the conversation
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    return messages


def bot(history: list):
    """Generate bot response"""
    if not history:
        return history

    # Convert history to LangChain format
    messages = format_chat_history(history)

    try:
        # Get response from LLM
        response, call = invoke.call(messages)
        chat_state.last_response = call

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


# Create Gradio interface
with gr.Blocks() as demo:
    with weave.attributes({"session": "local"}):
        gr.Markdown(
            """<div align="center"><h1>papersai</h1></div>""",
        )
        chatbot = gr.Chatbot(
            show_label=False,
            height=600,
            type="messages",
        )

        chat_input = gr.MultimodalTextbox(
            interactive=True,
            file_count="single",
            placeholder="Upload a document or type your message...",
            show_label=False,
        )

        # Set up the chat flow
        chat_msg = chat_input.submit(
            fn=add_message, inputs=[chatbot, chat_input], outputs=[chatbot, chat_input]
        )

        bot_msg = chat_msg.then(
            bot, inputs=chatbot, outputs=chatbot, api_name="bot_response"
        )

        # Like/dislike functionality
        chatbot.like(
            fn=print_like_dislike,
            inputs=None,
            outputs=None,
            like_user_message=True,
        )

if __name__ == "__main__":
    demo.launch(share=False)
