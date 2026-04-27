from __future__ import annotations

import streamlit as st
from google import genai
from google.genai import types

from prompt import SYSTEM_PROMPT

APP_TITLE = "Kitchen Assistant"
DEFAULT_MODEL = "gemini-2.5-flash"
INTRO_TEXT = (
    "This assistant helps you manage ingredients, get meal ideas, "
    "reduce food waste, and estimate basic nutrition."
)


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


@st.cache_resource
def get_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def build_contents(messages: list[dict[str, str]]) -> list[types.Content]:
    contents: list[types.Content] = []
    for message in messages:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part.from_text(text=message["content"])],
            )
        )
    return contents


def generate_reply(
    client: genai.Client,
    model_name: str,
    messages: list[dict[str, str]],
) -> str:
    response = client.models.generate_content(
        model=model_name,
        contents=build_contents(messages),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()


def clear_chat() -> None:
    st.session_state.messages = []


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🍽️", layout="centered")
    init_session_state()

    st.title(APP_TITLE)
    st.caption(INTRO_TEXT)

    with st.sidebar:
        st.subheader("About")
        st.write(
            "A lightweight Gemini-powered kitchen chatbot for ingredient tracking, "
            "meal ideas, and simple nutrition estimates."
        )

        st.subheader("How to use")
        st.write(
            "- Ask in English or Chinese.\n"
            "- Share your kitchen setup, pantry staples, dietary preferences, and current ingredients.\n"
            "- Ask what you can cook, what to buy, or for a quick meal nutrition estimate."
        )

        if st.button("Clear chat", use_container_width=True):
            clear_chat()
            st.rerun()

    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "Missing `GEMINI_API_KEY` in Streamlit Secrets. "
            "Add it before running the app."
        )
        st.stop()

    client = get_gemini_client(api_key)

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input("Ask about your ingredients, meals, or pantry...")
    if not user_input:
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply = generate_reply(
                    client=client,
                    model_name=DEFAULT_MODEL,
                    messages=st.session_state.messages,
                )
            except Exception as exc:
                reply = (
                    "Sorry, I couldn't reach Gemini just now. "
                    f"Please try again.\n\nError: `{exc}`"
                )

        if not reply:
            reply = "I don't have a response yet. Please try rephrasing that."

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
