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
KNOWLEDGE_PDF_FILES = [
    "DGA 美国居民膳食指南.pdf",
    "中国居民膳食指南(1).pdf",
]
NUTRITION_KEYWORDS = {
    "nutrition",
    "nutritional",
    "diet",
    "dietary",
    "calorie",
    "calories",
    "protein",
    "carb",
    "carbs",
    "fat",
    "fiber",
    "sugar",
    "sodium",
    "meal balance",
    "daily diet",
    "meal summary",
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
    "糖",
    "钠",
    "膳食",
    "健康",
    "总结今天",
}


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []


@st.cache_resource
def get_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def get_knowledge_status() -> dict[str, list[Path] | list[str]]:
    pdf_paths = [KNOWLEDGE_DIR / name for name in KNOWLEDGE_PDF_FILES]
    existing_pdfs = [path for path in pdf_paths if path.exists()]
    missing_pdfs = [path.name for path in pdf_paths if not path.exists()]
    txt_paths = sorted(KNOWLEDGE_DIR.glob("*.txt")) if KNOWLEDGE_DIR.exists() else []
    return {
        "pdfs": existing_pdfs,
        "missing_pdfs": missing_pdfs,
        "txts": txt_paths,
    }


def should_use_knowledge(user_input: str, messages: list[dict[str, str]]) -> bool:
    recent_text = " ".join(
        [message["content"] for message in messages[-4:] if message["role"] == "user"]
        + [user_input]
    ).lower()
    return any(keyword in recent_text for keyword in NUTRITION_KEYWORDS)


@st.cache_data(show_spinner=False)
def load_txt_knowledge(txt_paths: tuple[str, ...]) -> str:
    chunks: list[str] = []
    for path_str in txt_paths:
        path = Path(path_str)
        try:
            chunks.append(f"Reference summary from {path.name}:\n{path.read_text(encoding='utf-8')}")
        except UnicodeDecodeError:
            chunks.append(
                f"Reference summary from {path.name}:\n"
                f"{path.read_text(encoding='utf-8', errors='ignore')}"
            )
    return "\n\n".join(chunks).strip()


@st.cache_resource(show_spinner=False)
def upload_knowledge_pdfs(
    api_key: str, pdf_keys: tuple[tuple[str, int, int], ...]
) -> tuple[list[tuple[str, str, str]], list[str]]:
    client = genai.Client(api_key=api_key)
    uploaded_parts: list[tuple[str, str, str]] = []
    errors: list[str] = []

    # Knowledge PDFs are loaded from knowledge/ and uploaded once so they can be
    # attached to Gemini requests only when a nutrition-related question needs them.
    for path_str, _size, _mtime in pdf_keys:
        path = Path(path_str)
        try:
            uploaded_file = client.files.upload(file=str(path))
            uploaded_parts.append(
                (path.name, uploaded_file.uri, uploaded_file.mime_type or "application/pdf")
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    return uploaded_parts, errors


def get_knowledge_parts(api_key: str) -> tuple[list[types.Part], list[str], str]:
    status = get_knowledge_status()
    warnings: list[str] = []

    pdfs = status["pdfs"]
    if pdfs:
        pdf_keys = tuple(
            (str(path), path.stat().st_size, int(path.stat().st_mtime)) for path in pdfs
        )
        uploaded_parts, upload_errors = upload_knowledge_pdfs(api_key, pdf_keys)
        if uploaded_parts:
            parts = [
                types.Part.from_uri(file_uri=file_uri, mime_type=mime_type)
                for _name, file_uri, mime_type in uploaded_parts
            ]
            if upload_errors:
                warnings.extend(
                    [
                        "Some knowledge PDFs could not be attached; using the ones that uploaded successfully."
                    ]
                )
            return parts, warnings, "pdf"
        warnings.extend(
            [
                "Knowledge PDFs were found, but Gemini attachment failed. "
                "Trying local .txt knowledge summaries instead."
            ]
        )

    txt_paths = tuple(str(path) for path in status["txts"])
    if txt_paths:
        txt_content = load_txt_knowledge(txt_paths)
        if txt_content:
            return [types.Part.from_text(text=txt_content)], warnings, "txt"

    if pdfs:
        warnings.append(
            "No usable PDF attachment or .txt fallback is available for nutrition guidance."
        )
    return [], warnings, "none"


def build_contents(
    messages: list[dict[str, str]],
    knowledge_parts: list[types.Part] | None = None,
    knowledge_mode: str = "none",
) -> list[types.Content]:
    contents: list[types.Content] = []

    if knowledge_parts:
        knowledge_intro = (
            "Reference the attached dietary guidance documents when answering "
            "nutrition, calorie, meal-balance, daily diet summary, or health-related "
            "food questions. Use them as supporting knowledge, and say when advice is "
            "a general guideline rather than a precise medical instruction."
        )
        if knowledge_mode == "txt":
            knowledge_intro = (
                "Reference the attached text summaries from the dietary guidance files "
                "when answering nutrition, calorie, meal-balance, daily diet summary, "
                "or health-related food questions. Use them as supporting knowledge, "
                "and say when advice is a general guideline rather than a precise "
                "medical instruction."
            )
        contents.append(
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=knowledge_intro), *knowledge_parts],
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
    api_key: str,
    model_name: str,
    messages: list[dict[str, str]],
    user_input: str,
) -> str:
    knowledge_parts: list[types.Part] = []
    knowledge_mode = "none"
    if should_use_knowledge(user_input, messages):
        knowledge_parts, knowledge_warnings, knowledge_mode = get_knowledge_parts(api_key)
        if knowledge_warnings:
            st.session_state.knowledge_runtime_warnings = knowledge_warnings

    response = client.models.generate_content(
        model=model_name,
        contents=build_contents(messages, knowledge_parts=knowledge_parts, knowledge_mode=knowledge_mode),
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
    st.session_state.knowledge_runtime_warnings = []

    st.title(APP_TITLE)
    st.caption(INTRO_TEXT)

    knowledge_status = get_knowledge_status()

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

        if knowledge_status["missing_pdfs"]:
            st.warning(
                "Knowledge PDFs missing: " + ", ".join(knowledge_status["missing_pdfs"]),
                icon="⚠️",
            )

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
                    api_key=api_key,
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

    if st.session_state.knowledge_runtime_warnings:
        with st.sidebar:
            for warning in st.session_state.knowledge_runtime_warnings:
                st.caption(f"Knowledge note: {warning}")


if __name__ == "__main__":
    main()
