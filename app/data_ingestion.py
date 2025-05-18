import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()


def extract_text_from_pdf(pdf_path):
    """Extract raw text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def split_text(text, chunk_size=500, chunk_overlap=50):
    """Split text into chunks using LangChain splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def create_vectorstore(chunks, persist_dir="vectorstore/"):
    """Embed chunks and store them in a FAISS vector database."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(persist_dir)
    print(f"[INFO] Vectorstore saved to '{persist_dir}'")


if __name__ == "__main__":
    pdf_file = "data/sample.pdf"  # Change this as needed
    print(f"[INFO] Reading PDF: {pdf_file}")
    raw_text = extract_text_from_pdf(pdf_file)
    chunks = split_text(raw_text)
    print(f"[INFO] Total chunks created: {len(chunks)}")
    create_vectorstore(chunks)
