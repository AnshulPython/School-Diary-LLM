import os
import re
import streamlit as st
from groq import Groq
from pypdf import PdfReader

# Page setup
st.set_page_config(page_title="Document-Reader-LLM", layout="wide")
st.title("📄 Document-Reader-LLM")

# 1. Groq Authentication
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Add it in Streamlit Cloud -> Settings -> Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

# Dynamically fetch active Groq models to avoid decommission errors
@st.cache_data(ttl=3600)
def get_active_groq_models():
    try:
        model_list = client.models.list()
        # Filter for text chat models and sort them
        chat_models = [
            m.id for m in model_list.data 
            if not any(x in m.id.lower() for x in ["whisper", "vision", "guard", "embed"])
        ]
        return sorted(chat_models) if chat_models else ["llama-3.3-70b-versatile"]
    except Exception:
        return ["llama-3.3-70b-versatile"]

active_models = get_active_groq_models()

# 2. File Upload & Model Selector Sidebar
with st.sidebar:
    st.header("Document Control")
    uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    st.divider()
    st.header("Model Settings")
    selected_model = st.selectbox("Active Groq Model", options=active_models)

def extract_pdf_pages(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages

def find_relevant_pages(pages, query):
    keywords = [word.lower() for word in re.findall(r"\w+", query) if len(word) > 2]
    if not keywords:
        return pages[:3]
    
    scored_pages = []
    for item in pages:
        body = item["text"].lower()
        score = sum(body.count(kw) for kw in keywords)
        scored_pages.append((score, item))
    
    scored_pages.sort(key=lambda entry: entry[0], reverse=True)
    return [entry[1] for entry in scored_pages[:3]]

if uploaded_file and "document_pages" not in st.session_state:
    with st.spinner("Parsing document..."):
        st.session_state.document_pages = extract_pdf_pages(uploaded_file)
        st.success(f"Document ingested: {len(st.session_state.document_pages)} pages processed!")

# 3. Query & Interaction
user_prompt = st.text_input("Ask any question regarding the uploaded document (supports any language):")

if user_prompt:
    if "document_pages" not in st.session_state:
        st.warning("Please upload a PDF document in the sidebar first.")
    else:
        matched_pages = find_relevant_pages(st.session_state.document_pages, user_prompt)
        document_context = "\n---\n".join([f"[Page {p['page']}]:\n{p['text']}" for p in matched_pages])

        system_prompt = f"""You are Document-Reader-LLM, a precise and factual document query assistant.
Answer the user's question using ONLY the provided document excerpts.
- Answer in the EXACT language used by the user.
- Explicitly reference the page numbers containing the facts you cite.
- If the document excerpts do not contain sufficient evidence to answer, state: "The uploaded document does not provide information on this topic." Do not fabricate facts.

Document Excerpts:
{document_context}"""

        with st.spinner(f"Analyzing document with {selected_model}..."):
            completion = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1
            )

            st.markdown("### Response")
            st.write(completion.choices[0].message.content)

            with st.expander("View Referenced Excerpts"):
                for entry in matched_pages:
                    st.markdown(f"**Page {entry['page']} excerpt:**")
                    st.text(entry["text"][:600] + ("..." if len(entry["text"]) > 600 else ""))
                    
