import os
import sys
import subprocess

# 1. THE AUTO-INSTALLER HACK
# This intercepts the missing module error and forces the server to install 
# the required packages on the fly before running the rest of the script.
try:
    from groq import Groq
except ModuleNotFoundError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", 
        "groq", "langchain", "langchain-community", 
        "langchain-text-splitters", "pypdf", "faiss-cpu", "sentence-transformers"
    ])
    from groq import Groq

# 2. STANDARD IMPORTS
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.set_page_config(page_title="School Policy Agent", layout="wide")
st.title("🏫 School Handbook & Policy Agent")

# 3. API KEY AUTHENTICATION
groq_api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    st.error("Missing GROQ_API_KEY. Please add it in Streamlit Advanced Settings -> Secrets.")
    st.stop()

client = Groq(api_key=groq_api_key)

# 4. LIGHTWEIGHT EMBEDDING MODEL
@st.cache_resource
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

embedding_model = get_embedding_model()

# 5. SIDEBAR: PDF INGESTION
with st.sidebar:
    st.header("Admin Settings")
    uploaded_file = st.file_uploader("Upload School Handbook (PDF)", type=["pdf"])

if uploaded_file and "vector_db" not in st.session_state:
    with st.spinner("Processing PDF and indexing sections (this takes a few seconds)..."):
        with open("uploaded_handbook.pdf", "wb") as f:
            f.write(uploaded_file.get_buffer())
        
        loader = PyPDFLoader("uploaded_handbook.pdf")
        documents = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
        docs = splitter.split_documents(documents)
        
        st.session_state.vector_db = FAISS.from_documents(docs, embedding_model)
        st.success(f"Handbook loaded successfully ({len(docs)} segments indexed)!")

# 6. MAIN CHAT INTERFACE
user_query = st.text_input("Ask a question about rules, schedules, or dress codes (in any language):")

if user_query:
    if "vector_db" not in st.session_state:
        st.warning("Please upload a handbook PDF in the sidebar first.")
    else:
        # Retrieve the most relevant contextual chunks
        retriever = st.session_state.vector_db.as_retriever(search_kwargs={"k": 3})
        matches = retriever.invoke(user_query)
        context = "\n---\n".join([doc.page_content for doc in matches])

        system_prompt = f"""You are an accurate, helpful school administrative assistant.
Answer the user's question based strictly on the policy excerpt provided below.
- Reply in the EXACT same language the user writes in.
- Cite specific rules or page numbers when available.
- If the excerpt does NOT contain the answer, reply strictly: "This information is not covered in the current school policy document." Do not invent rules.

Policy Excerpt:
{context}"""

        with st.spinner("Analyzing policies..."):
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
                for i, doc in enumerate(matches, 1):
                    st.markdown(f"**Source Section {i} (Page {doc.metadata.get('page', 'N/A')}):**")
                    st.write(doc.page_content)
                
