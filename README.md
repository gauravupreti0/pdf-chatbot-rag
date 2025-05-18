## PDF Chatbot (RAG-Based, Free & Offline-Compatible)

An intelligent chatbot that lets users **conversationally query PDFs** using **Retrieval-Augmented Generation (RAG)** — without relying on OpenAI. Built with **LangChain**, **FAISS**, and **HuggingFace** models (local or hosted), this system is ideal for **fully private**, **API-free**, and **scalable AI document search**.

---

### Features

- Upload any PDF and chat with its contents
- Uses **semantic search** to retrieve the most relevant chunks
- Answers powered by HuggingFace models (e.g., Zephyr, Mistral)
- Streamlit UI for PDF upload + chat interface
- CLI mode also supported for dev or automation
- Works with **free APIs** or **fully offline LLMs (via `llama-cpp`)**
- Built with modular, production-ready Python components

---

### Tech Stack

| Layer            | Tools Used                                        |
| ---------------- | ------------------------------------------------- |
| **LLM**          | HuggingFaceHub (`zephyr`, `mistral`), `llama-cpp` |
| **Embeddings**   | `sentence-transformers/all-MiniLM-L6-v2` (HF)     |
| **Vector Store** | FAISS (local)                                     |
| **Retrieval**    | LangChain `RetrievalQA`                           |
| **UI**           | Streamlit                                         |
| **PDF Parser**   | PyPDF2                                            |

---

### Project Structure

```bash
pdf-chatbot-rag/
├── app/
│   ├── data_ingestion.py       # PDF → text → chunks → vectorstore
│   ├── rag_pipeline.py         # Load FAISS + build RAG chain
│   └── utils.py                # Chunking, text utilities
├── data/                       # Uploaded PDFs
├── vectorstore/                # FAISS DB files
├── models/                     # (Optional) .gguf files for llama-cpp
├── web_ui/
│   └── chatbot_ui.py           # Streamlit UI
├── run.py                      # CLI chatbot (terminal)
├── requirements.txt            # Dependencies
├── .env                        # Environment variables
└── README.md
```

---

### Setup & Usage

#### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Also install `llama-cpp-python` if running locally:

```bash
pip install llama-cpp-python
```

---

#### 2. Environment Variables

Create a `.env` file:

```env
HUGGINGFACEHUB_API_TOKEN=your_huggingface_token  # if using HuggingFaceHub
```

---

#### 3. Run the app

##### Streamlit UI

```bash
streamlit run web_ui/chatbot_ui.py
```

##### CLI

```bash
python run.py
```

---

### Supported Models

| Mode          | Model/Tool                            | Notes                        |
| ------------- | ------------------------------------- | ---------------------------- |
| Online (free) | `HuggingFaceH4/zephyr-7b-beta`        | Good balance, API hosted     |
| Offline       | `mistral-7b-instruct` via `llama-cpp` | Full local inference (.gguf) |
| Embeddings    | `all-MiniLM-L6-v2` (HF Transformers)  | Fast + effective             |

---

### Security

- All processing is done **locally**
- No OpenAI / ChatGPT usage
- FAISS and pickle deserialization are **guarded with `allow_dangerous_deserialization=True`** (only safe when you control the data)

---

### Coming Soon / Enhancements

- Multiple PDF document handling
- Chat memory support
- Show source chunks with each response
- Dockerfile for containerized deployment
- Offline LLM with `Ollama` or `GPT4All`

---

### Credits

- [LangChain](https://github.com/langchain-ai/langchain)
- [Hugging Face Transformers](https://huggingface.co/models)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [FAISS](https://github.com/facebookresearch/faiss)
