from __future__ import annotations

from pathlib import Path

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
KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
GUIDELINE_SPECS = [
    {
        "key": "us_guidelines",
        "label": "U.S. Dietary Guidelines",
        "filename": "us_dietary_guidelines.txt",
    },
    {
        "key": "cn_guidelines",
        "label": "Chinese Dietary Guidelines",
        "filename": "chinese_dietary_guidelines.txt",
    },
]
NUTRITION_KEYWORDS = {
    "nutrition",
    "nutritional",
    "diet",
    "dietary",
    "calorie",
    "calories",
    "meal balance",
    "daily diet",
    "daily summary",
    "meal summary",
    "healthy eating",
    "weight loss",
    "fat loss",
    "protein",
    "carbohydrate",
    "carbohydrates",
    "carb",
    "carbs",
    "fat",
    "fiber",
    "vegetable",
    "vegetables",
    "fruit",
    "sodium",
    "sugar",
    "oil",
    "health",
    "healthy",
    "减脂",
    "增肌",
    "控糖",
    "饮食",
    "营养",
    "热量",
    "卡路里",
    "蛋白质",
    "碳水",
    "脂肪",
    "纤维",
    "蔬菜",
    "水果",
    "钠",
    "糖",
    "油",
    "膳食",
    "健康",
    "总结今天",
}


def detect_message_language(text: str) -> str:
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return "zh"
    return "en"


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_texts" not in st.session_state:
        st.session_state.knowledge_texts = {}
    if "knowledge_status" not in st.session_state:
        st.session_state.knowledge_status = {}
    if "knowledge_last_request_used" not in st.session_state:
        st.session_state.knowledge_last_request_used = []
    if "conversation_language" not in st.session_state:
        st.session_state.conversation_language = None


@st.cache_resource
def get_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def knowledge_path(filename: str) -> Path:
    # Local guideline text files are loaded from the knowledge/ folder.
    return KNOWLEDGE_DIR / filename


@st.cache_data(show_spinner=False)
def read_knowledge_text(path_str: str) -> str:
    path = Path(path_str)
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def load_knowledge_texts() -> None:
    for spec in GUIDELINE_SPECS:
        path = knowledge_path(spec["filename"])
        if not path.exists():
            st.session_state.knowledge_status[spec["key"]] = {
                "label": spec["label"],
                "status": "missing",
                "error": f"Missing {spec['filename']}",
            }
            st.session_state.knowledge_texts[spec["key"]] = ""
            continue

        text = read_knowledge_text(str(path))
        if text:
            st.session_state.knowledge_status[spec["key"]] = {
                "label": spec["label"],
                "status": "loaded",
                "error": "",
            }
            st.session_state.knowledge_texts[spec["key"]] = text
        else:
            st.session_state.knowledge_status[spec["key"]] = {
                "label": spec["label"],
                "status": "empty",
                "error": f"{spec['filename']} is empty",
            }
            st.session_state.knowledge_texts[spec["key"]] = ""


def should_use_knowledge(user_input: str, messages: list[dict[str, str]]) -> bool:
    recent_user_text = " ".join(
        [message["content"] for message in messages[-4:] if message["role"] == "user"]
        + [user_input]
    ).lower()
    return any(keyword in recent_user_text for keyword in NUTRITION_KEYWORDS)


def get_knowledge_context() -> tuple[str, list[str]]:
    sections: list[str] = []
    used_labels: list[str] = []

    for spec in GUIDELINE_SPECS:
        status = st.session_state.knowledge_status.get(spec["key"], {})
        text = st.session_state.knowledge_texts.get(spec["key"], "")
        if status.get("status") == "loaded" and text:
            sections.append(f"{spec['label']} reference:\n{text}")
            used_labels.append(spec["label"])

    return "\n\n".join(sections).strip(), used_labels


def build_contents(
    messages: list[dict[str, str]],
    knowledge_text: str = "",
    conversation_language: str | None = None,
) -> list[types.Content]:
    contents: list[types.Content] = []

    if conversation_language == "zh":
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "For this conversation, use Chinese as the primary reply language. "
                            "If you need to ask onboarding questions, ask them in Chinese."
                        )
                    )
                ],
            )
        )
    elif conversation_language == "en":
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "For this conversation, use English as the primary reply language. "
                            "If you need to ask onboarding questions, ask them in English."
                        )
                    )
                ],
            )
        )

    if knowledge_text:
        # Nutrition-related requests attach the local .txt guideline references here.
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Use the following U.S. Dietary Guidelines and Chinese "
                            "Dietary Guidelines text references when relevant. If the "
                            "references do not contain enough information, say so "
                            "instead of inventing details.\n\n"
                            f"{knowledge_text}"
                        )
                    )
                ],
            )
        )

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
    user_input: str,
) -> str:
    knowledge_text = ""
    st.session_state.knowledge_last_request_used = []
    conversation_language = st.session_state.conversation_language
    if not conversation_language:
        conversation_language = detect_message_language(user_input)
        st.session_state.conversation_language = conversation_language

    if should_use_knowledge(user_input, messages):
        knowledge_text, used_labels = get_knowledge_context()
        st.session_state.knowledge_last_request_used = used_labels

    response = client.models.generate_content(
        model=model_name,
        contents=build_contents(
            messages,
            knowledge_text=knowledge_text,
            conversation_language=conversation_language,
        ),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.conversation_language = None


def render_sidebar() -> None:
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

        st.subheader("Knowledge Status")
        for spec in GUIDELINE_SPECS:
            status = st.session_state.knowledge_status.get(spec["key"], {})
            st.caption(f"{spec['label']}: {status.get('status', 'missing')}")
            if status.get("error"):
                st.warning(f"{spec['label']}: {status['error']}", icon="⚠️")

        if st.session_state.knowledge_last_request_used:
            st.caption(
                "Used in latest nutrition request: "
                + ", ".join(st.session_state.knowledge_last_request_used)
            )
        else:
            st.caption("Guideline text files are attached only for nutrition-related questions.")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🍽️", layout="centered")
    init_session_state()
    load_knowledge_texts()

    st.title(APP_TITLE)
    st.caption(INTRO_TEXT)

    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "Missing `GEMINI_API_KEY` in Streamlit Secrets. "
            "Add it before running the app."
        )
        st.stop()

    client = get_gemini_client(api_key)
    render_sidebar()

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
                    user_input=user_input,
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
