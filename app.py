from __future__ import annotations

from pathlib import Path
import re

import streamlit as st
from google import genai
from google.genai import types

from prompt import SYSTEM_PROMPT

APP_TITLE = "Kitchen Assistant"
DEFAULT_MODEL = "gemini-2.5-flash-lite"
INTRO_TEXT = (
    "Manage ingredients, get meal ideas, reduce food waste, "
    "and estimate basic nutrition."
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
APP_CSS = """
<style>
:root {
    --bg: #F5FAF8;
    --sidebar-bg: #DCEFE7;
    --primary: #4E9F8A;
    --secondary: #F2C14E;
    --text: #24312F;
    --card: #FFFFFF;
    --user-bubble: #FFF1C7;
    --assistant-bubble: #DCEFE7;
    --border: rgba(36, 49, 47, 0.12);
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stAppViewContainer"] {
    background: var(--bg);
}

[data-testid="stSidebar"] {
    background: var(--sidebar-bg);
    border-right: 1px solid rgba(78, 159, 138, 0.12);
}

[data-testid="stSidebar"] > div:first-child {
    background: var(--sidebar-bg);
}

[data-testid="block-container"] {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 46rem;
}

h1, h2, h3, p, label, div, span {
    color: var(--text);
}

.hero-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1rem 1.1rem;
    margin: 0.35rem 0 1.4rem;
    box-shadow: 0 10px 30px rgba(78, 159, 138, 0.06);
}

.hero-card p {
    margin: 0;
    font-size: 1rem;
    line-height: 1.6;
}

.sidebar-card {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid rgba(78, 159, 138, 0.12);
    border-radius: 18px;
    padding: 0.95rem 1rem;
    margin-top: 0.8rem;
    box-shadow: 0 10px 24px rgba(78, 159, 138, 0.06);
}

.sidebar-card-title {
    font-weight: 700;
    margin-bottom: 0.65rem;
}

[data-testid="stButton"] > button {
    background: var(--primary);
    color: white;
    border: none;
    border-radius: 999px;
    padding: 0.65rem 1rem;
    font-weight: 600;
    box-shadow: 0 8px 20px rgba(78, 159, 138, 0.18);
}

[data-testid="stButton"] > button:hover {
    background: #438a77;
}

[data-testid="stChatInput"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 0.2rem 0.4rem;
    box-shadow: 0 8px 24px rgba(36, 49, 47, 0.05);
}

[data-testid="stChatInput"] textarea {
    color: var(--text);
}

[data-testid="stChatMessage"] {
    padding: 0.2rem 0;
}

[data-testid="stChatMessage"] > div {
    border-radius: 20px;
}

[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
    border-radius: 18px;
    padding: 0.95rem 1rem;
    line-height: 1.6;
    box-shadow: 0 6px 20px rgba(36, 49, 47, 0.04);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) > div {
    background: var(--user-bubble);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid="stMarkdownContainer"] {
    background: var(--user-bubble);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) > div {
    background: var(--assistant-bubble);
}

[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid="stMarkdownContainer"] {
    background: var(--assistant-bubble);
}

[data-testid="chatAvatarIcon-user"] {
    background: #F2C14E !important;
    color: #24312F !important;
}

[data-testid="chatAvatarIcon-assistant"] {
    background: #4E9F8A !important;
    color: #ffffff !important;
}

[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.9);
    border: 1px solid var(--border);
    border-radius: 18px;
    overflow: hidden;
}

.developer-note {
    margin-top: 1.4rem;
}

@media (max-width: 768px) {
    [data-testid="block-container"] {
        padding-top: 1rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .hero-card {
        margin-bottom: 1rem;
    }
}
</style>
"""


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
    if "last_usage" not in st.session_state:
        st.session_state.last_usage = {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
        }
    if "session_usage" not in st.session_state:
        st.session_state.session_usage = {
            "prompt_tokens": 0,
            "response_tokens": 0,
            "total_tokens": 0,
        }


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
) -> tuple[str, dict[str, int]]:
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

    usage_metadata = getattr(response, "usage_metadata", None)
    usage = {
        "prompt_tokens": int(getattr(usage_metadata, "prompt_token_count", 0) or 0),
        "response_tokens": int(getattr(usage_metadata, "candidates_token_count", 0) or 0),
        "total_tokens": int(getattr(usage_metadata, "total_token_count", 0) or 0),
    }
    return (response.text or "").strip(), usage


def format_api_error(exc: Exception) -> str:
    raw_error = str(exc)
    if "429" in raw_error and "RESOURCE_EXHAUSTED" in raw_error:
        retry_match = re.search(r"retry in ([0-9.]+)s", raw_error, re.IGNORECASE)
        retry_seconds = None
        if retry_match:
            retry_seconds = max(1, round(float(retry_match.group(1))))

        if retry_seconds is not None:
            return (
                f"Rate limit reached for Gemini right now. Please wait about "
                f"{retry_seconds} seconds and try again."
            )
        return "Rate limit reached for Gemini right now. Please wait a bit and try again."

    return f"Sorry, I couldn't reach Gemini just now. Please try again.\n\nError: `{raw_error}`"


def update_usage_state(usage: dict[str, int]) -> None:
    st.session_state.last_usage = usage
    st.session_state.session_usage = {
        "prompt_tokens": st.session_state.session_usage["prompt_tokens"] + usage["prompt_tokens"],
        "response_tokens": st.session_state.session_usage["response_tokens"] + usage["response_tokens"],
        "total_tokens": st.session_state.session_usage["total_tokens"] + usage["total_tokens"],
    }


def clear_chat() -> None:
    st.session_state.messages = []
    st.session_state.conversation_language = None
    st.session_state.last_usage = {
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0,
    }
    st.session_state.session_usage = {
        "prompt_tokens": 0,
        "response_tokens": 0,
        "total_tokens": 0,
    }


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

        knowledge_lines: list[str] = []
        for spec in GUIDELINE_SPECS:
            status = st.session_state.knowledge_status.get(spec["key"], {})
            status_text = status.get("status", "missing")
            knowledge_lines.append(f"<p>{spec['label']}: {status_text}</p>")
            if status.get("error"):
                knowledge_lines.append(f"<p>{spec['label']}: {status['error']}</p>")

        if st.session_state.knowledge_last_request_used:
            knowledge_lines.append(
                "<p>Used in latest nutrition request: "
                + ", ".join(st.session_state.knowledge_last_request_used)
                + "</p>"
            )
        else:
            knowledge_lines.append(
                "<p>Guideline text files are attached only for nutrition-related questions.</p>"
            )

        st.markdown(
            '<div class="sidebar-card">'
            '<div class="sidebar-card-title">Knowledge Status</div>'
            + "".join(knowledge_lines)
            + "</div>",
            unsafe_allow_html=True,
        )


def render_usage_footer() -> None:
    last_usage = st.session_state.last_usage
    session_usage = st.session_state.session_usage

    st.markdown('<div class="developer-note">', unsafe_allow_html=True)
    with st.expander("Developer info", expanded=False):
        st.caption(
            "Token usage"
            f" | Last request: prompt {last_usage['prompt_tokens']}, "
            f"response {last_usage['response_tokens']}, total {last_usage['total_tokens']}"
        )
        st.caption(
            f"Session total: prompt {session_usage['prompt_tokens']}, "
            f"response {session_usage['response_tokens']}, total {session_usage['total_tokens']}"
        )
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🍽️", layout="centered")
    init_session_state()
    load_knowledge_texts()
    st.markdown(APP_CSS, unsafe_allow_html=True)

    st.title("🍳 Kitchen Assistant")
    st.markdown(
        f'<div class="hero-card"><p>{INTRO_TEXT}</p></div>',
        unsafe_allow_html=True,
    )

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
        render_usage_footer()
        return

    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                reply, usage = generate_reply(
                    client=client,
                    model_name=DEFAULT_MODEL,
                    messages=st.session_state.messages,
                    user_input=user_input,
                )
                update_usage_state(usage)
            except Exception as exc:
                reply = format_api_error(exc)

        if not reply:
            reply = "I don't have a response yet. Please try rephrasing that."

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    render_usage_footer()


if __name__ == "__main__":
    main()
