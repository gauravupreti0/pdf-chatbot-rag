from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(text, chunk_size=500, chunk_overlap=50):
    """Split large text into smaller overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_text(text)


def print_colored(text, color="cyan"):
    """Print text with optional color (requires termcolor)."""
    try:
        from termcolor import colored

        print(colored(text, color))
    except ImportError:
        print(text)
