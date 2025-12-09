import os
import re

# Make Streamlit behave nicely on Windows (optional, harmless elsewhere)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup
from huggingface_hub import InferenceClient


# ---------------- Streamlit page config ---------------- #

st.set_page_config(
    page_title="LLM Document App",
    page_icon="🤖",
    layout="wide",
)


# ---------------- LLM loading (Hugging Face Inference API) ---------------- #

@st.cache_resource
def load_llm():
    """
    Create an InferenceClient that calls an instruction-tuned model
    on the Hugging Face Inference API.

    You must define HF_TOKEN in Streamlit Cloud secrets:
        HF_TOKEN = "hf_XXXXXXXXXXXXXXXXXXXXXXXX"
    """
    # Get token from Streamlit secrets (cloud) or environment (local)
    try:
        hf_token = st.secrets["HF_TOKEN"]
    except Exception:
        hf_token = os.environ.get("HF_TOKEN")

    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. "
            "In Streamlit Cloud, go to Settings → Secrets and add HF_TOKEN."
        )

    # Use a public text-generation model
    model_name = "HuggingFaceH4/zephyr-7b-beta"

    client = InferenceClient(model=model_name, token=hf_token)
    return client


# ---------------- File reading helpers ---------------- #

def read_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def read_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join(p.text for p in doc.paragraphs)


def read_txt(uploaded_file):
    data = uploaded_file.read()
    try:
        return data.decode("utf-8")
    except Exception:
        return data.decode("latin-1", errors="ignore")


def read_html(uploaded_file):
    data = uploaded_file.read()
    try:
        html = data.decode("utf-8", errors="ignore")
    except Exception:
        html = str(data)
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text(separator="\n")


def extract_text_from_file(uploaded_file):
    """
    Detect file type by extension and read text accordingly.
    """
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    if name.endswith(".docx"):
        return read_docx(uploaded_file)
    if name.endswith(".txt"):
        return read_txt(uploaded_file)
    if name.endswith(".htm") or name.endswith(".html"):
        return read_html(uploaded_file)

    # Fallback: try treating it as text
    return read_txt(uploaded_file)


# ---------------- Post-processing for abbreviation mode ---------------- #

def clean_abbrev_answer(raw_answer):
    """
    Take the raw LLM output and keep only lines of the form:
        ABBR: full term

    Returns a list of cleaned "ABBR: full term" strings.
    """
    clean_lines = []

    for line in raw_answer.splitlines():
        line = line.strip()
        if not line:
            continue

        # Pattern: ABBR: full term
        match = re.match(r"^([A-Z][A-Z0-9]{1,10})\s*:\s*(.+)$", line)
        if match:
            abbr = match.group(1).strip()
            full = match.group(2).strip()
            clean_lines.append(f"{abbr}: {full}")

    return clean_lines


# ---------------- Main app ---------------- #

def main():
    st.title("Input to AI")
    st.write(
        "Ask a question and optionally upload documents. "
        "The model will answer based on your input."
    )

    st.markdown("---")

    # User input
    user_question = st.text_area(
        "Your question:",
        height=120,
        placeholder=(
            "Example for abbreviations:\n"
            "Extract all full term (ABBREVIATION) pairs from the document. "
            "List them, one per line, using the format ABBREVIATION: full term."
        ),
    )

    uploaded_files = st.file_uploader(
        "Upload files (optional):",
        type=["pdf", "docx", "txt", "htm", "html"],
        accept_multiple_files=True,
    )

    # Decide whether we are in abbreviation mode
    abbreviation_mode = "abbreviation" in user_question.lower()

    if abbreviation_mode:
        st.markdown("### AI Response (Abbreviation Index):")
    else:
        st.markdown("### AI Response:")

    if st.button("Get Answer"):
        if not user_question.strip():
            st.warning("Please enter a question first.")
            return

        # Load model
        try:
            llm = load_llm()
        except Exception as e:
            st.error("Error loading the model:")
            st.exception(e)
            return

        # -------------- ABBREVIATION MODE -------------- #
        if abbreviation_mode and uploaded_files:
            for uploaded in uploaded_files:
                document_text = extract_text_from_file(uploaded) or ""

                # To avoid sending extremely long documents, truncate
                max_chars = 8000
                truncated_text = document_text[:max_chars]

                prompt = (
                    "You are an AI that extracts abbreviation definitions from academic text.\n\n"
                    "TASK:\n"
                    "From the document text below, extract all pairs where an abbreviation is "
                    "defined by its full term. Return ONLY one pair per line, using the format:\n"
                    "ABBREVIATION: full term\n\n"
                    "Do not include any explanations, notes, or extra text. Do not add bullet points.\n\n"
                    f"DOCUMENT TEXT:\n{truncated_text}\n\n"
                    "Now return the abbreviation index:"
                )

                try:
                    raw_answer = llm.text_generation(
                        prompt,
                        max_new_tokens=256,
                        temperature=0.0,
                    )
                    raw_answer = raw_answer.strip()
                except Exception as e:
                    st.error("Error calling the model:")
                    st.exception(e)
                    return

                abbrev_lines = clean_abbrev_answer(raw_answer)

                st.markdown(f"#### File: `{uploaded.name}`")
                if not abbrev_lines:
                    st.markdown("_No abbreviations found._")
                else:
                    for line in abbrev_lines:
                        st.write(line)

            return  # Done in abbreviation mode

        # -------------- GENERAL QA MODE -------------- #

        # Collect document text (if any)
        combined_text = ""
        if uploaded_files:
            all_texts = []
            for uploaded in uploaded_files:
                text = extract_text_from_file(uploaded)
                if text:
                    all_texts.append(f"--- File: {uploaded.name} ---\n{text}")
            combined_text = "\n\n".join(all_texts)

        try:
            if combined_text:
                # Limit how much text we send
                max_chars = 12000
                truncated = combined_text[:max_chars]

                prompt = (
                    "You are a helpful assistant answering questions about documents.\n\n"
                    f"USER QUESTION:\n{user_question}\n\n"
                    "DOCUMENTS:\n"
                    f"{truncated}\n\n"
                    "Use the documents above to answer the question as clearly as possible. "
                    "If something is not in the documents, you may answer from general knowledge."
                )
            else:
                # No documents uploaded – general Q&A
                prompt = (
                    "You are a helpful assistant.\n\n"
                    f"USER QUESTION:\n{user_question}\n\n"
                    "Answer clearly and concisely."
                )

            raw_answer = llm.text_generation(
                prompt,
                max_new_tokens=512,
                temperature=0.2,
            )
            answer = raw_answer.strip()

            st.subheader("AI Response:")
            st.write(answer)

            if combined_text:
                with st.expander("Show combined document text (truncated)"):
                    st.text(combined_text[:4000])

        except Exception as e:
            st.error("Error running the model:")
            st.exception(e)


if __name__ == "__main__":
    main()
