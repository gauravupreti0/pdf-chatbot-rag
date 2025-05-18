from app.rag_pipeline import load_vectorstore, create_qa_chain
from utils.utils import print_colored


def main():
    print_colored("\n📄 PDF Chatbot (RAG-based)", "cyan")
    print_colored("Type your questions below. Type 'exit' to quit.\n", "yellow")

    # Load the FAISS vectorstore and set up QA pipeline
    try:
        vectorstore = load_vectorstore()
        qa_chain = create_qa_chain(vectorstore)
    except Exception as e:
        print_colored(f"[ERROR] Failed to load vectorstore: {e}", "red")
        return

    while True:
        query = input("🧑‍💻 You: ")
        if query.lower() in ["exit", "quit", "q"]:
            print_colored("👋 Exiting chatbot. Goodbye!", "green")
            break

        try:
            response = qa_chain.run(query)
            print_colored(f"🤖 Bot: {response}\n", "blue")
        except Exception as err:
            print_colored(f"[ERROR] Something went wrong: {err}", "red")


if __name__ == "__main__":
    main()
