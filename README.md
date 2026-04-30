# Kitchen Assistant

A minimal Streamlit chatbot for a Gemini-based kitchen assistant.  
Access the site here: [Kitchen Assistant](https://ai-kitchen-assistant-agent.streamlit.app/)

## Features

- Streamlit chat UI with `st.chat_input()` and `st.chat_message()`
- Gemini API integration via `google-genai`
- Session-based conversation memory with `st.session_state`
- Bilingual behavior based on the user's language
- Sidebar with app info, quick instructions, and a clear-chat action

## Setup

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your Streamlit secret:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

For local development, place it in `.streamlit/secrets.toml`.

## Run locally

```bash
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repository to GitHub.
2. Create a new Streamlit Community Cloud app pointing to `app.py`.
3. In the app settings, add this secret:

```toml
GEMINI_API_KEY = "your-api-key-here"
```

4. Deploy.

## Notes

- The API key is loaded from `st.secrets["GEMINI_API_KEY"]`.
- The default model is `gemini-2.5-flash-lite`.
- No database, login, long-term storage, or external memory layer is included yet.
- Nutrition and diet-related questions now use local text knowledge files from `knowledge/`.
- Expected text files are `us_dietary_guidelines.txt` and `chinese_dietary_guidelines.txt`.
- The source PDFs can still remain in `knowledge/`, but the app currently reads the `.txt` versions for the most reliable behavior.
