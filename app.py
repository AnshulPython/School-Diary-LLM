import os
import re
import streamlit as st
from groq import Groq
from pypdf import PdfReader

st.set_page_config(page_title="School Policy Agent", layout="wide")
st.title("🏫 School Handbook & Policy Agent")

# 1. API Key Auth
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Please add it in Streamlit Advanced Settings -> Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

# 2. Sidebar Upload
with st.sidebar:
    st.header("Admin Settings")
    uploaded_file = st.file_uploader("Upload School Handbook (PDF)", type=["pdf"])

def extract_handbook_pages(pdf_file):
    reader = PdfReader(pdf_file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages

def score_pages(pages, query):
    words = [w.lower() for w in re.findall(r'\w+', query) if len(w) > 2]
    if not words:
        return pages[:3]
    
    scored = []
    for p in pages:
        page_text = p["text"].lower()
        score = sum(page_text.count(word) for word in words)
        scored.append((score, p))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:3]]

if uploaded_file and "handbook_pages" not in st.session_state:
    with st.spinner("Reading and preparing handbook..."):
        st.session_state.handbook_pages = extract_handbook_pages(uploaded_file)
        st.success(f"Handbook loaded: {len(st.session_state.handbook_pages)} pages processed!")

# 3. Chat Interface
user_query = st.text_input("Ask a question about rules, schedules, or dress codes (in any language):")

if user_query:
    if "handbook_pages" not in st.session_state:
        st.warning("Please upload a school handbook PDF in the sidebar first.")
    else:
        relevant_pages = score_pages(st.session_state.handbook_pages, user_query)
        context = "\n---\n".join([f"[Page {p['page']}]:\n{p['text']}" for p in relevant_pages])

        system_prompt = f"""You are an accurate, helpful school administrative assistant.
Answer the user's question based strictly on the policy excerpt provided below.
- Reply in the EXACT same language the user writes in.
- Cite specific rules and page numbers when available.
- If the excerpt does NOT contain the answer, reply strictly: "This information is not covered in the current school policy document." Do not invent rules.

Policy Excerpt:
{context}"""

        with st.spinner("Finding answer..."):
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ],
                temperature=0.1
            )

            st.markdown("### Answer")
            st.write(response.choices[0].message.content)

            with st.expander("View Cited Excerpts"):
                for p in relevant_pages:
                    st.markdown(f"**Page {p['page']}:**")
                    st.text(p["text"][:600] + "...")
                    
