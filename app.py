import os
import re
import streamlit as st
from groq import Groq
from pypdf import PdfReader

st.set_page_config(page_title="Document-Reader-LLM", layout="wide")
st.title("📄 Document-Reader-LLM")

# 1. API & Admin Authentication
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
admin_password = st.secrets.get("ADMIN_PASSWORD", "admin123")

if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Add it in Streamlit Cloud -> Settings -> Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

# Helpers to persist and load model choice from disk
MODEL_FILE = "selected_model.txt"
PREFERRED_DEFAULT = "openai/gpt-oss-20b"

def get_saved_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "r") as f:
            return f.read().strip()
    return PREFERRED_DEFAULT

def save_model_choice(model_name):
    with open(MODEL_FILE, "w") as f:
        f.write(model_name)

# Dynamically fetch ALL active text chat models on Groq without excluding any families
@st.cache_data(ttl=3600)
def get_active_groq_models():
    try:
        model_list = client.models.list()
        chat_models = [
            m.id for m in model_list.data 
            if not any(x in m.id.lower() for x in ["whisper", "vision", "guard", "embed"])
        ]
        return sorted(chat_models) if chat_models else [PREFERRED_DEFAULT]
    except Exception:
        return [PREFERRED_DEFAULT]

active_models = get_active_groq_models()

# Ensure preferred model is in list; default gracefully if not
saved_model = get_saved_model()
default_model = saved_model if saved_model in active_models else (
    PREFERRED_DEFAULT if PREFERRED_DEFAULT in active_models else active_models[0]
)

if "current_model" not in st.session_state:
    st.session_state.current_model = default_model

def extract_pdf_pages(file_path_or_buffer):
    reader = PdfReader(file_path_or_buffer)
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

@st.cache_resource
def load_persisted_document():
    if os.path.exists("indexed_document.pdf"):
        return extract_pdf_pages("indexed_document.pdf")
    return []

# 2. Sidebar: Admin Controls Only
with st.sidebar:
    st.header("Admin Access")
    entered_password = st.text_input("Enter Admin Password", type="password")

    if entered_password == admin_password:
        st.success("Admin unlocked")
        
        # Display full list of models
        current_index = active_models.index(st.session_state.current_model) if st.session_state.current_model in active_models else 0
        new_model_choice = st.selectbox(
            "Select Active Groq Model",
            options=active_models,
            index=current_index
        )
        
        # Save choice to disk whenever updated
        if new_model_choice != st.session_state.current_model:
            st.session_state.current_model = new_model_choice
            save_model_choice(new_model_choice)
            st.toast(f"Model updated to {new_model_choice}!")

        uploaded_file = st.file_uploader("Upload / Replace PDF Document", type=["pdf"])
        if uploaded_file:
            if st.button("Index and Save for All Users"):
                with st.spinner("Processing and saving document..."):
                    with open("indexed_document.pdf", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.cache_resource.clear()
                    st.rerun()
    elif entered_password:
        st.error("Incorrect password.")
    else:
        st.caption("🔒 Model & document controls are restricted to the admin.")

# 3. User Search Interface
user_prompt = st.text_input("Ask any question regarding the document (supports any language):")

if user_prompt:
    active_doc = load_persisted_document()
    
    if not active_doc:
        st.warning("No document has been loaded by the administrator yet.")
    else:
        matched_pages = find_relevant_pages(active_doc, user_prompt)
        document_context = "\n---\n".join([f"[Page {p['page']}]:\n{p['text']}" for p in matched_pages])

        system_prompt = f"""You are Document-Reader-LLM, a precise and factual document query assistant.
Answer the user's question using ONLY the provided document excerpts.
- Answer in the EXACT language used by the user.
- Explicitly reference the page numbers containing the facts you cite.
- If the document excerpts do not contain sufficient evidence to answer, state: "The uploaded document does not provide information on this topic." Do not fabricate facts.

Document Excerpts:
{document_context}"""

        with st.spinner(f"Searching document using {st.session_state.current_model}..."):
            completion = client.chat.completions.create(
                model=st.session_state.current_model,
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
    
