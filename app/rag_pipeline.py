from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from dotenv import load_dotenv

load_dotenv()


def load_vectorstore(persist_dir="vectorstore/"):
    """Load FAISS vectorstore from disk."""
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.load_local(
        persist_dir,
        embeddings,
        allow_dangerous_deserialization=True,  # ✅ Safe if you created it
    )


def create_qa_chain(vectorstore):
    llm = LlamaCpp(
        model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        temperature=0.3,
        max_tokens=256,
        n_ctx=2048,
        verbose=False,
    )
    retriever = vectorstore.as_retriever()
    return RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
