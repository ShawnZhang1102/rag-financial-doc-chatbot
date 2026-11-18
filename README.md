# Financial Document Chatbot (RAG)

> Class Project · Johns Hopkins University

A retrieval-augmented generation (RAG) chatbot that answers questions about financial documents. Upload PDFs — research papers, resumes, filings — and the app chunks and embeds them into a FAISS vector store, retrieves the most relevant passages for each question, and answers with a large language model.

## Features

- **Chat with PDFs**: ask questions grounded in the documents you upload, instead of relying only on the model's training data.
- **Local or remote LLMs**: run fully locally with Ollama (Mistral + `nomic-embed-text`), or use Google Gemini / OpenAI.
- **Conversation history**: variants with and without chat history.
- **Sample documents included**: synthetic resumes and a research paper under `PDFs/` for testing.

## Tech Stack

Python · LangChain · FAISS · Streamlit · Ollama · Google Gemini · OpenAI

## How to Run

```bash
# 1. Create the environment
conda create -n chatbot python=3.11
conda activate chatbot

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key as an environment variable (remote models only)
export GOOGLE_API_KEY="your-key-here"   # for Gemini
export OPENAI_API_KEY="your-key-here"   # for OpenAI

# 4. Run one of the apps
streamlit run chat_with_pdf_ollama.py            # PDF chat, local Ollama
streamlit run chat_with_pdf_ollama_with_history.py
streamlit run chat_with_pdf_gemini.py            # PDF chat, Gemini
streamlit run chat_with_pdf_gemini_with_history.py
streamlit run chat_with_pdf_openai.py            # PDF chat, OpenAI
streamlit run chat_with_local_ollama.py          # plain chat, local Ollama
streamlit run chat_with_gemini.py                # plain chat, Gemini
```

For local models, install [Ollama](https://ollama.com/download) and pull the models first: `ollama pull mistral` and `ollama pull nomic-embed-text`. `test_with_ollama.py` checks that the local model responds.

## Files

| File | What it does |
| --- | --- |
| `chat_with_pdf_ollama.py` | PDF Q&A with local Ollama |
| `chat_with_pdf_ollama_with_history.py` | PDF Q&A with local Ollama + chat history |
| `chat_with_pdf_gemini.py` | PDF Q&A with Google Gemini |
| `chat_with_pdf_gemini_with_history.py` | PDF Q&A with Gemini + chat history |
| `chat_with_pdf_openai.py` | PDF Q&A with OpenAI |
| `chat_with_local_ollama.py` | Plain chat with local Ollama |
| `chat_with_gemini.py` | Plain chat with Gemini |
| `test_with_ollama.py` | Smoke test for the local Ollama model |
| `requirements.txt` | Python dependencies |
| `PDFs/` | Sample documents for testing |

## Notes

- API keys are read from environment variables. Never hardcode keys into the code.
- The sample PDFs are synthetic resumes and a published paper, included only as test data.
