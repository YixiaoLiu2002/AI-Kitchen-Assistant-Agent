from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import streamlit as st
from google import genai
from google.genai import types
from pypdf import PdfReader

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
PDF_READY_TIMEOUT_SECONDS = 45
PDF_READY_POLL_SECONDS = 2
PDF_FALLBACK_MAX_PAGES = 24
PDF_FALLBACK_MAX_CHARS_PER_FILE = 18000


def init_knowledge_state() -> None:
    if "knowledge_runtime_warnings" not in st.session_state:
        st.session_state.knowledge_runtime_warnings = []
    if "knowledge_runtime_status" not in st.session_state:
        st.session_state.knowledge_runtime_status = {
            "last_triggered": False,
            "last_mode": "not-used",
            "loaded_files": [],
            "fallback_files": [],
            "errors": [],
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


def update_knowledge_runtime_status(
    *,
    triggered: bool,
    mode: str,
    loaded_files: list[str] | None = None,
    fallback_files: list[str] | None = None,
    errors: list[str] | None = None,
) -> None:
    st.session_state.knowledge_runtime_status = {
        "last_triggered": triggered,
        "last_mode": mode,
        "loaded_files": loaded_files or [],
        "fallback_files": fallback_files or [],
        "errors": errors or [],
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


@st.cache_data(show_spinner=False)
def extract_pdf_text_to_txt(
    pdf_path_str: str,
    file_size: int,
    file_mtime: int,
) -> tuple[str | None, str | None]:
    del file_size, file_mtime
    pdf_path = Path(pdf_path_str)
    txt_path = pdf_path.with_suffix(".txt")

    try:
        reader = PdfReader(str(pdf_path))
        chunks: list[str] = []
        total_chars = 0

        # Fallback path: if direct PDF upload is unavailable, extract a concise
        # local text summary from the PDF and save it beside the source file.
        for page in reader.pages[:PDF_FALLBACK_MAX_PAGES]:
            page_text = (page.extract_text() or "").strip()
            if not page_text:
                continue

            remaining = PDF_FALLBACK_MAX_CHARS_PER_FILE - total_chars
            if remaining <= 0:
                break

            page_excerpt = page_text[:remaining]
            chunks.append(page_excerpt)
            total_chars += len(page_excerpt)

        if not chunks:
            return None, f"{pdf_path.name}: no extractable text found"

        summary_text = (
            f"Extracted reference text from {pdf_path.name} "
            f"(first {min(len(reader.pages), PDF_FALLBACK_MAX_PAGES)} pages, truncated):\n\n"
            + "\n\n".join(chunks)
        )
        txt_path.write_text(summary_text, encoding="utf-8")
        return str(txt_path), None
    except Exception as exc:
        return None, f"{pdf_path.name}: text extraction failed ({exc})"


def wait_for_uploaded_file_ready(client: genai.Client, uploaded_file: Any) -> Any:
    deadline = time.time() + PDF_READY_TIMEOUT_SECONDS
    current_file = uploaded_file

    while time.time() < deadline:
        state_name = getattr(getattr(current_file, "state", None), "name", None)
        if state_name in {None, "ACTIVE"}:
            return current_file
        if state_name == "FAILED":
            raise RuntimeError(f"file processing failed for {current_file.name}")
        time.sleep(PDF_READY_POLL_SECONDS)
        current_file = client.files.get(name=current_file.name)

    raise TimeoutError(f"timed out waiting for {current_file.name} to become ACTIVE")


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
            uploaded_file = wait_for_uploaded_file_ready(client, uploaded_file)
            uploaded_parts.append(
                (path.name, uploaded_file.uri, uploaded_file.mime_type or "application/pdf")
            )
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    return uploaded_parts, errors


def get_knowledge_parts(api_key: str) -> tuple[list[types.Part], list[str], str]:
    status = get_knowledge_status()
    warnings: list[str] = []
    load_errors: list[str] = []
    fallback_files: list[str] = []

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
                load_errors.extend(upload_errors)
                warnings.append(
                    "Some knowledge PDFs could not be attached; using the ones that uploaded successfully."
                )
            update_knowledge_runtime_status(
                triggered=True,
                mode="pdf",
                loaded_files=[name for name, _uri, _mime in uploaded_parts],
                errors=load_errors,
            )
            return parts, warnings, "pdf"
        load_errors.extend(upload_errors)
        warnings.append(
            "Knowledge PDFs were found, but Gemini attachment failed. Trying local .txt knowledge summaries instead."
        )

    txt_paths_list = [str(path) for path in status["txts"]]
    if not txt_paths_list and pdfs:
        for path in pdfs:
            txt_path_str, extraction_error = extract_pdf_text_to_txt(
                str(path), path.stat().st_size, int(path.stat().st_mtime)
            )
            if txt_path_str:
                txt_paths_list.append(txt_path_str)
                fallback_files.append(Path(txt_path_str).name)
            elif extraction_error:
                load_errors.append(extraction_error)

    txt_paths = tuple(txt_paths_list)
    if txt_paths:
        txt_content = load_txt_knowledge(txt_paths)
        if txt_content:
            if not fallback_files:
                fallback_files = [Path(path).name for path in txt_paths]
            update_knowledge_runtime_status(
                triggered=True,
                mode="txt",
                fallback_files=fallback_files,
                errors=load_errors,
            )
            return [types.Part.from_text(text=txt_content)], warnings, "txt"

    if pdfs:
        warnings.append(
            "No usable PDF attachment or .txt fallback is available for nutrition guidance."
        )
    update_knowledge_runtime_status(
        triggered=True,
        mode="failed",
        errors=load_errors,
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
    else:
        update_knowledge_runtime_status(triggered=False, mode="not-used")

    response = client.models.generate_content(
        model=model_name,
        contents=build_contents(
            messages,
            knowledge_parts=knowledge_parts,
            knowledge_mode=knowledge_mode,
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


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🍽️", layout="centered")
    init_session_state()
    init_knowledge_state()
    st.session_state.knowledge_runtime_warnings = []

    st.title(APP_TITLE)
    st.caption(INTRO_TEXT)

    knowledge_status = get_knowledge_status()
    knowledge_runtime_status = st.session_state.knowledge_runtime_status

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
        st.caption(
            f"PDFs found: {len(knowledge_status['pdfs'])}/{len(KNOWLEDGE_PDF_FILES)}"
        )
        for pdf_name in KNOWLEDGE_PDF_FILES:
            found = pdf_name in [path.name for path in knowledge_status["pdfs"]]
            st.caption(f"{'Found' if found else 'Missing'}: {pdf_name}")

        if knowledge_runtime_status["last_mode"] == "pdf":
            st.success(
                "Dietary PDFs were successfully attached to the latest nutrition-related Gemini request."
            )
        elif knowledge_runtime_status["last_mode"] == "txt":
            st.info(
                "PDF attachment was not used for the latest nutrition-related request. Text fallback was loaded instead."
            )
        elif knowledge_runtime_status["last_mode"] == "failed":
            st.warning(
                "A nutrition-related request tried to load dietary knowledge, but neither PDF attachment nor text fallback succeeded."
            )
        else:
            st.caption(
                "The dietary knowledge files are only loaded for nutrition-related questions."
            )

        if knowledge_runtime_status["loaded_files"]:
            st.caption(
                "Attached to Gemini: "
                + ", ".join(knowledge_runtime_status["loaded_files"])
            )
        if knowledge_runtime_status["fallback_files"]:
            st.caption(
                "Fallback text loaded: "
                + ", ".join(knowledge_runtime_status["fallback_files"])
            )
        if knowledge_runtime_status["errors"]:
            st.caption(
                "Load notes: " + " | ".join(knowledge_runtime_status["errors"][:2])
            )

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
