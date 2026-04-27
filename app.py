from __future__ import annotations

import shutil
import time
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
        "filename": "us_dietary_guidelines.pdf",
    },
    {
        "key": "cn_guidelines",
        "label": "Chinese Dietary Guidelines",
        "filename": "chinese_dietary_guidelines.pdf",
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
FILE_READY_TIMEOUT_SECONDS = 90
FILE_READY_POLL_SECONDS = 2
UPLOAD_TMP_DIR = KNOWLEDGE_DIR / ".upload_tmp"


def init_session_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_files" not in st.session_state:
        st.session_state.knowledge_files = {}
    if "knowledge_bootstrapped" not in st.session_state:
        st.session_state.knowledge_bootstrapped = False
    if "knowledge_last_request_used" not in st.session_state:
        st.session_state.knowledge_last_request_used = []
    if "knowledge_last_request_errors" not in st.session_state:
        st.session_state.knowledge_last_request_errors = []


@st.cache_resource
def get_gemini_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def guideline_path(filename: str) -> Path:
    # Local PDF files are loaded from the knowledge/ folder.
    return KNOWLEDGE_DIR / filename


def ascii_upload_path(spec: dict[str, str]) -> Path:
    suffix = guideline_path(spec["filename"]).suffix or ".pdf"
    return UPLOAD_TMP_DIR / f"{spec['key']}{suffix}"


def default_knowledge_entry(spec: dict[str, str]) -> dict[str, str]:
    path = guideline_path(spec["filename"])
    return {
        "label": spec["label"],
        "filename": spec["filename"],
        "path": str(path),
        "status": "missing" if not path.exists() else "not uploaded",
        "error": "",
        "file_name": "",
        "file_uri": "",
        "mime_type": "application/pdf",
    }


def sync_missing_statuses() -> None:
    for spec in GUIDELINE_SPECS:
        entry = st.session_state.knowledge_files.get(
            spec["key"], default_knowledge_entry(spec)
        )
        path = guideline_path(spec["filename"])
        if not path.exists():
            entry.update(
                {
                    "status": "missing",
                    "error": "Local PDF file not found.",
                    "file_name": "",
                    "file_uri": "",
                }
            )
        elif entry["status"] == "missing":
            entry.update({"status": "not uploaded", "error": ""})
        st.session_state.knowledge_files[spec["key"]] = entry


def wait_for_file_active(client: genai.Client, file_name: str) -> object:
    deadline = time.time() + FILE_READY_TIMEOUT_SECONDS
    last_file = client.files.get(name=file_name)

    while time.time() < deadline:
        state_name = getattr(getattr(last_file, "state", None), "name", None)
        if state_name in {None, "ACTIVE"}:
            return last_file
        if state_name == "FAILED":
            raise RuntimeError(f"{file_name} failed during Gemini file processing.")
        time.sleep(FILE_READY_POLL_SECONDS)
        last_file = client.files.get(name=file_name)

    raise TimeoutError(f"{file_name} did not become ACTIVE before timeout.")


def prepare_ascii_upload_copy(spec: dict[str, str]) -> Path:
    source_path = guideline_path(spec["filename"])
    UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)
    temp_path = ascii_upload_path(spec)
    shutil.copyfile(source_path, temp_path)
    return temp_path


def upload_guideline_file(client: genai.Client, spec: dict[str, str]) -> dict[str, str]:
    path = guideline_path(spec["filename"])
    entry = default_knowledge_entry(spec)
    if not path.exists():
        entry["error"] = "Local PDF file not found."
        return entry

    try:
        # The PDF is uploaded to Gemini Files API once and its reference is saved
        # in session state so we do not re-upload on every chat turn.
        safe_upload_path = prepare_ascii_upload_copy(spec)
        uploaded_file = client.files.upload(file=str(safe_upload_path))
        active_file = wait_for_file_active(client, uploaded_file.name)
        entry.update(
            {
                "status": "loaded",
                "error": "",
                "file_name": active_file.name,
                "file_uri": active_file.uri,
                "mime_type": active_file.mime_type or "application/pdf",
            }
        )
    except Exception as exc:
        entry.update(
            {
                "status": "upload failed",
                "error": str(exc),
                "file_name": "",
                "file_uri": "",
            }
        )
    return entry


def ensure_guideline_loaded(
    client: genai.Client,
    spec: dict[str, str],
    *,
    force_reupload: bool = False,
) -> dict[str, str]:
    current = st.session_state.knowledge_files.get(
        spec["key"], default_knowledge_entry(spec)
    )
    path = guideline_path(spec["filename"])
    if not path.exists():
        current.update(
            {
                "status": "missing",
                "error": "Local PDF file not found.",
                "file_name": "",
                "file_uri": "",
            }
        )
        st.session_state.knowledge_files[spec["key"]] = current
        return current

    if not force_reupload and current.get("file_name"):
        try:
            active_file = wait_for_file_active(client, current["file_name"])
            current.update(
                {
                    "status": "loaded",
                    "error": "",
                    "file_name": active_file.name,
                    "file_uri": active_file.uri,
                    "mime_type": active_file.mime_type or "application/pdf",
                }
            )
            st.session_state.knowledge_files[spec["key"]] = current
            return current
        except Exception:
            pass

    updated = upload_guideline_file(client, spec)
    st.session_state.knowledge_files[spec["key"]] = updated
    return updated


def bootstrap_knowledge_files(client: genai.Client) -> None:
    sync_missing_statuses()
    if st.session_state.knowledge_bootstrapped:
        return
    for spec in GUIDELINE_SPECS:
        ensure_guideline_loaded(client, spec)
    st.session_state.knowledge_bootstrapped = True


def should_use_knowledge(user_input: str, messages: list[dict[str, str]]) -> bool:
    recent_user_text = " ".join(
        [message["content"] for message in messages[-4:] if message["role"] == "user"]
        + [user_input]
    ).lower()
    return any(keyword in recent_user_text for keyword in NUTRITION_KEYWORDS)


def get_attached_guideline_parts(client: genai.Client) -> list[types.Part]:
    parts: list[types.Part] = []
    used_labels: list[str] = []
    errors: list[str] = []

    for spec in GUIDELINE_SPECS:
        entry = ensure_guideline_loaded(client, spec)
        if entry["status"] == "loaded" and entry["file_uri"] and entry["mime_type"]:
            try:
                parts.append(
                    types.Part.from_uri(
                        file_uri=entry["file_uri"],
                        mime_type=entry["mime_type"],
                    )
                )
                used_labels.append(entry["label"])
            except Exception as exc:
                errors.append(f"{entry['label']}: {exc}")
        else:
            errors.append(f"{entry['label']}: {entry['status']}")

    st.session_state.knowledge_last_request_used = used_labels
    st.session_state.knowledge_last_request_errors = errors
    return parts


def build_contents(
    messages: list[dict[str, str]],
    knowledge_parts: list[types.Part] | None = None,
) -> list[types.Content]:
    contents: list[types.Content] = []

    if knowledge_parts:
        # When nutrition-related requests need guideline support, the uploaded PDF
        # references are attached directly to the Gemini request here.
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(
                        text=(
                            "Use the attached U.S. Dietary Guidelines and Chinese "
                            "Dietary Guidelines as reference documents when relevant. "
                            "If the documents do not contain enough information, say so "
                            "instead of inventing details."
                        )
                    ),
                    *knowledge_parts,
                ],
            )
        )

    for message in messages:
        if message["role"] == "assistant":
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=message["content"])],
                )
            )
        else:
            contents.append(
                types.Content(
                    role="user",
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
    knowledge_parts: list[types.Part] = []
    st.session_state.knowledge_last_request_used = []
    st.session_state.knowledge_last_request_errors = []

    if should_use_knowledge(user_input, messages):
        knowledge_parts = get_attached_guideline_parts(client)

    response = client.models.generate_content(
        model=model_name,
        contents=build_contents(messages, knowledge_parts=knowledge_parts),
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return (response.text or "").strip()


def clear_chat() -> None:
    st.session_state.messages = []


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
            entry = st.session_state.knowledge_files.get(
                spec["key"], default_knowledge_entry(spec)
            )
            st.caption(f"{entry['label']}: {entry['status']}")
            if entry["status"] in {"missing", "upload failed"} and entry["error"]:
                st.warning(f"{entry['label']}: {entry['error']}", icon="⚠️")

        if st.session_state.knowledge_last_request_used:
            st.caption(
                "Used in latest nutrition request: "
                + ", ".join(st.session_state.knowledge_last_request_used)
            )
        elif st.session_state.knowledge_last_request_errors:
            st.caption(
                "Latest nutrition request could not use: "
                + " | ".join(st.session_state.knowledge_last_request_errors)
            )
        else:
            st.caption(
                "Guideline PDFs are attached only for nutrition-related questions."
            )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="🍽️", layout="centered")
    init_session_state()

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
    bootstrap_knowledge_files(client)
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
