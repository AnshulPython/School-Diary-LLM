import os
import re
import streamlit as st
from groq import Groq
from pypdf import PdfReader

st.set_page_config(page_title="Document-Reader-LLM", layout="wide")
st.title("📄 Document-Reader-LLM")

# 1. API Key & Admin Auth
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
admin_password = st.secrets.get("ADMIN_PASSWORD", "admin123")

if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Add it in Streamlit Cloud -> Settings -> Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

# Persistent global storage for the document and default model
if "current_model" not in st.session_state:
    st.session_state.current_model = "llama-3.3-70b-versatile"
if "document_pages" not in st.session_state:
    st.session_state.document_pages = []

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

# 2. Sidebar: Hidden Behind Admin Login
with st.sidebar:
    st.header("Admin Controls")
    entered_password = st.text_input("Enter Admin Password", type="password")

    if entered_password == admin_password:
        st.success("Admin mode unlocked")
        
        # Admin can switch the active model
        model_choice = st.selectbox(
            "Select Model",
            options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
            index=0
        )
        st.session_state.current_model = model_choice

        # Admin can upload or replace the PDF
        uploaded_file = st.file_uploader("Upload / Replace PDF", type=["pdf"])
        if uploaded_file:
            if st.button("Index and Save Document"):
                with st.spinner("Extracting and saving document..."):
                    st.session_state.document_pages = extract_pdf_pages(uploaded_file)
                    st.success(f"Indexed {len(st.session_state.document_pages)} pages!")
    elif entered_password:
        st.error("Incorrect password.")
    else:
        st.info("Log in with the admin password to upload documents or change the model.")

# 3. Public User Search Interface
user_prompt = st.text_input("Ask a question about the document (in any language):")

if user_prompt:
    if not st.session_state.document_pages:
        st.warning("No document has been uploaded or indexed yet. Please check back later.")
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

        with st.spinner("Searching document..."):
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
                    
