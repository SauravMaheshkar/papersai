# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "gradio",
#     "openai",
#     "pymupdf4llm",
#     "python-dotenv",
#     "weave",
# ]
# ///
import time

import gradio as gr
import pymupdf4llm
import weave
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = weave.init("rag-cookbook")

global call_id
call_id = None

# System prompt
CHAT_SYSTEM_PROMPT: str = (
    "You are an expert in the field of AI and Machine Learning."
    "You are chatting with a researcher who is conducting research "
    "based on the following paper. You are expected to provide "
    "insightful responses to the researcher's queries.\n"
    "\n"
    "\n"
    "{context_str}\n"
    "\n"
    "\n"
    "Researcher:"
)


def log_user_feedback(x: gr.LikeData):
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
    call = client.get_call(call_id)

    # Remove any existing feedback before adding new feedback
    for existing_feedback in list(call.feedback):
        call.feedback.purge(existing_feedback.id)

    if x.liked:
        call.feedback.add_reaction("👍")
    else:
        call.feedback.add_reaction("👎")


def add_message(history, message):
    """
    Add user message to history and parse PDF to markdown.

    Args:
        history(list(dict[str, str])): Chat history
        message(dict[str, str]): User message, including text and files

    Returns:
        Updated chat history and a multimodal textbox
    """
    # Parse PDF to markdown and add to system prompt
    for x in message["files"]:
        history.append(
            {
                "role": "system",
                "content": CHAT_SYSTEM_PROMPT.format(
                    context_str=pymupdf4llm.to_markdown(x, show_progress=False)
                ),
            }
        )

    # Add user message to history
    if message["text"] is not None:
        history.append({"role": "user", "content": message["text"]})
    return history, gr.MultimodalTextbox(value=None, interactive=False)


def bot_response(history: list):
    """
    Get response from the LLM model and stream it to the chatbot.

    Args:
        history(list(dict[str, str])): Chat history

    Yields:
        response from the LLM
    """
    global call_id

    @weave.op()
    def llm_response(messages):
        llm = OpenAI()
        response = llm.chat.completions.create(model="gpt-4o", messages=messages)
        response = str(response.choices[0].message.content)
        return response

    # Get LLM response
    with weave.attributes({"env": "local"}):
        response, call = llm_response.call(history)
        call_id = call.id

    # Add empty assistant message
    history.append({"role": "assistant", "content": ""})

    # Stream the response
    for character in response:
        history[-1]["content"] += character
        time.sleep(0.02)
        yield history


with gr.Blocks() as demo:
    # Initialize chatbot and chat input
    chatbot = gr.Chatbot(elem_id="chatbot", bubble_full_width=False, type="messages")
    chat_input = gr.MultimodalTextbox(
        interactive=True,
        placeholder="Enter message or upload file...",
        show_label=False,
    )

    # Add user message to chat history
    chat_msg = chat_input.submit(
        fn=add_message, inputs=[chatbot, chat_input], outputs=[chatbot, chat_input]
    )

    # Get response from the LLM
    bot_msg = chat_msg.then(
        fn=bot_response, inputs=[chatbot], outputs=[chatbot], api_name="bot_response"
    )
    bot_msg.then(
        fn=lambda: gr.MultimodalTextbox(interactive=True),
        inputs=None,
        outputs=[chat_input],
    )

    # Record user feedback on assistant's response
    chatbot.like(
        fn=log_user_feedback, inputs=None, outputs=None, like_user_message=True
    )

demo.launch()
