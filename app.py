import os
import re  # for cleaning abbreviation output

# Make Streamlit + torch behave nicely on Windows
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
from pypdf import PdfReader
import docx
from bs4 import BeautifulSoup
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# ------------- Streamlit page config ------------- #

st.set_page_config(
    page_title="LLM Document App",
    page_icon="🤖",
    layout="wide",
)

# ------------- LLM loading (Hugging Face) ------------- #

@st.cache_resource
def load_llm():
    """
    Load a small, instruction-tuned open-source model from Hugging Face.
    The model weights and cache will go to F:\\LLM_CACHE (because of the
    environment variables you already set: HF_HOME / TRANSFORMERS_CACHE / TORCH_HOME).
    """
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype="auto",
    )

    gen = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,  # more deterministic / less random
    )
    return gen

# ------------- File reading helpers ------------- #

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
    soup = BeautifulSoup(data.decode("utf-8", errors="ignore"), "html.parser")
    return soup.get_text(separator="\n")

def extract_text_from_file(uploaded_file):
    """
    Detect file type and return plain text.
    """
    name = uploaded_file.name.lower()
    uploaded_file.seek(0)

    if name.endswith(".pdf"):
        return read_pdf(uploaded_file)
    if name.endswith(".docx"):
        return read_docx(uploaded_file)
    if name.endswith(".txt"):
        return read_txt(uploaded_file)
    if name.endswith(".html") or name.endswith(".htm"):
        return read_html(uploaded_file)

    return ""

# ------------- Helper: clean abbreviation output ------------- #

def clean_abbrev_answer(raw_answer: str) -> list[str]:
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
        # pattern: ABBR: full term
        m = re.match(r'^([A-Z][A-Z0-9]{1,10})\s*:\s*(.+)$', line)
        if m:
            abbr = m.group(1).strip()
            full = m.group(2).strip()
            clean_lines.append(f"{abbr}: {full}")
    return clean_lines

# ------------- Main app ------------- #

def main():
    st.title("Input to AI")
    st.write(
        "Ask a question and optionally upload documents. "
        "The model will answer based on your input."
    )

    st.markdown("---")

    question = st.text_area("Your question:", height=120)

    uploaded_files = st.file_uploader(
        "Upload files (optional):",
        type=["pdf", "docx", "txt", "html", "htm"],
        accept_multiple_files=True,
    )

    if st.button("Get Answer", type="primary"):
        if not question.strip():
            st.warning("Please type a question first.")
            return

        lower_q = question.lower()
        is_abbrev_task = (
            "abbreviation" in lower_q
            or "abbreviations" in lower_q
            or "abbrev" in lower_q
        )

        with st.spinner("Thinking..."):
            llm = load_llm()

            # ---------- SPECIAL MODE: Abbreviation index per article ---------- #
            if uploaded_files and is_abbrev_task:
                st.subheader("AI Response (Abbreviation Index):")

                for uploaded in uploaded_files:
                    doc_text = extract_text_from_file(uploaded)
                    if not doc_text:
                        continue

                    # Limit context length for tiny model
                    max_chars = 6000
                    if len(doc_text) > max_chars:
                        doc_text = doc_text[:max_chars]

                    prompt = (
                        "You are an assistant that EXTRACTS ABBREVIATIONS "
                        "from scientific documents.\n"
                        "Your task is to build an abbreviation index from the DOCUMENT below.\n"
                        "An abbreviation index is a list like:\n"
                        "WDC: weighted degree centrality\n"
                        "SH: structural holes\n"
                        "ERGM: exponential random graph model\n"
                        "\n"
                        "Rules:\n"
                        "- Only include abbreviations that actually appear in the document.\n"
                        "- Most abbreviations appear in the form 'full term (ABBR)'. Use that pattern.\n"
                        "- The left side must be just the abbreviation token (e.g., 'WDC', 'ERGM', 'CAS').\n"
                        "- The right side must be the full term from the document, written clearly.\n"
                        "- Do NOT invent or guess abbreviations not supported by the text.\n"
                        "- Do NOT add explanations, bullets, or extra sentences.\n"
                        "- Output ONLY the index, one abbreviation per line, in the format 'ABBR: full term'.\n"
                        "\n"
                        "[DOCUMENT]\n"
                        f"{doc_text}\n\n"
                        "[ANSWER]\n"
                    )

                    result = llm(prompt)[0]["generated_text"]

                    # Strip prompt part
                    if "[ANSWER]" in result:
                        answer = result.split("[ANSWER]", 1)[-1].strip()
                    elif "Answer:" in result:
                        answer = result.split("Answer:", 1)[-1].strip()
                    else:
                        answer = result.strip()

                    # Clean to only ABBR: full term lines
                    abbrev_lines = clean_abbrev_answer(answer)

                    st.markdown(f"### File: `{uploaded.name}`")
                    if not abbrev_lines:
                        st.markdown("_No abbreviations found._")
                    else:
                        # Format like bullet list: - **WDC**: weighted degree centrality
                        bullets = []
                        for line in abbrev_lines:
                            abbr, full = line.split(":", 1)
                            bullets.append(f"- **{abbr.strip()}**: {full.strip()}")
                        st.markdown("\n".join(bullets))

                # We’re done in abbreviation mode
                return

            # ---------- NORMAL QA MODES (combined docs or no docs) ---------- #

            # Collect text from all uploaded documents (if any)
            doc_texts = []
            if uploaded_files:
                for f in uploaded_files:
                    text = extract_text_from_file(f)
                    if text:
                        doc_texts.append(text)

            document_text = "\n\n".join(doc_texts)

            # Limit context length
            max_chars = 6000
            if len(document_text) > max_chars:
                document_text = document_text[:max_chars]

            # Build the prompt for non-abbreviation tasks
            if document_text:
                prompt = (
                    "You are a helpful assistant.\n"
                    "Use ONLY the document text below to answer the question.\n"
                    "If the answer is not in the document, say exactly:\n"
                    "'I am not sure based on the document.'\n"
                    "Use clear, short sentences.\n\n"
                    "[DOCUMENT]\n"
                    f"{document_text}\n\n"
                    "[QUESTION]\n"
                    f"{question}\n\n"
                    "[ANSWER]"
                )
            else:
                # No document, general QA
                prompt = (
                    "You are a helpful assistant.\n"
                    "Answer the question clearly and stay on topic.\n\n"
                    f"Question: {question}\n"
                    "Answer:"
                )

            try:
                result = llm(prompt)[0]["generated_text"]

                # Try to remove the prompt part from the output
                if "[ANSWER]" in result:
                    answer = result.split("[ANSWER]", 1)[-1].strip()
                elif "Answer:" in result:
                    answer = result.split("Answer:", 1)[-1].strip()
                else:
                    answer = result.strip()

                st.subheader("AI Response:")
                st.code(answer, language="text")

                if document_text:
                    with st.expander("Show document text the model saw"):
                        st.text(document_text)

            except Exception as e:
                st.error(f"Error running the model: {e}")


if __name__ == "__main__":
    main()
