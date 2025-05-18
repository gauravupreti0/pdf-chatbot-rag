import os, sys
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.data_ingestion import extract_text_from_pdf, split_text, create_vectorstore
from app.rag_pipeline import load_vectorstore, create_qa_chain

# Streamlit settings
st.set_page_config(page_title="PDF Chatbot (RAG)", layout="centered")

st.title("📄 PDF Chatbot (RAG-based)")
st.markdown("Upload a PDF, and ask questions about its content.")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

# Temporary directory to store uploaded PDF
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# Process and load PDF
if uploaded_file is not None:
    file_path = os.path.join(DATA_DIR, uploaded_file.name)

    # Save file
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ PDF uploaded successfully!")

    # Ingest document
    with st.spinner("🔄 Processing PDF and creating vectorstore..."):
        raw_text = extract_text_from_pdf(file_path)
        chunks = split_text(raw_text)
        create_vectorstore(chunks)
        st.success("✅ Vectorstore created successfully!")

    # Chat interface
    st.markdown("---")
    st.subheader("💬 Ask a question from the PDF")

    query = st.text_input("Enter your question:")
    if query:
        with st.spinner("🤖 Generating answer..."):
            vectorstore = load_vectorstore()
            qa_chain = create_qa_chain(vectorstore)
            response = qa_chain.run(query)
            st.success("✅ Answer generated!")
            st.write(f"**Answer:** {response}")
