import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import tempfile
import time
import google.generativeai as genai
from google.api_core import retry


# Set Google API key (replace with your key or use an env variable)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "") # "YOUR_GOOGLE_API_KEY"  # Replace with your actual Gemini API key
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


st.set_page_config(page_title="Chat with Your PDFs (Gemini)")

st.title("📄💬 Chat with Your PDFs (Gemini)")

# File uploader
uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    documents = []

    current_file_names = sorted([f.name for f in uploaded_files])

    # Process PDFs if vector stores don't exist or file list has changed
    if "vector_stores" not in st.session_state or st.session_state.get("uploaded_file_names") != current_file_names:
        with st.spinner("Processing your PDFs..."):
            embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            # ------------------------------------------------- #
            # Build a separate vector store per uploaded file
            def embed_with_retry(chunks, embeddings, retries=5):
                for attempt in range(retries):
                    try:
                        return FAISS.from_documents(chunks, embeddings)
                    except Exception as e:
                        if attempt < retries - 1 and ("503" in str(e) or "unavailable" in str(e).lower()):
                            wait = 2 ** attempt
                            st.warning(f"Embedding service unavailable, retrying in {wait}s... (attempt {attempt+1}/{retries})")
                            time.sleep(wait)
                        else:
                            raise

            per_doc_stores = {}
            with tempfile.TemporaryDirectory() as temp_dir:
                for file in uploaded_files:
                    temp_file_path = os.path.join(temp_dir, file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(file.getbuffer())
                    loader = PyPDFLoader(temp_file_path)
                    file_docs = loader.load()
                    chunks = text_splitter.split_documents(file_docs)
                    per_doc_stores[file.name] = embed_with_retry(chunks, embeddings)

            st.session_state.vector_stores = per_doc_stores
            st.session_state.uploaded_file_names = current_file_names
            # ------------------------------------------------- #


        st.success("✅ PDFs uploaded and processed! You can now start chatting.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            content = message["content"].replace("$", r"\$") if message["role"] == "assistant" else message["content"]
            st.markdown(content)

    # User input
    user_input = st.chat_input("Ask a question about your PDFs...")

    if user_input:
        # Immediately add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display the user message immediately
        with st.chat_message("user"):
            st.markdown(user_input)



        # Retrieve top chunks from each document separately, then combine
        def multi_doc_retriever(query):
            all_chunks = []
            for store in st.session_state.vector_stores.values():
                results = store.similarity_search(query, k=10)
                all_chunks.extend(results)
            return all_chunks

        # Create a custom prompt template
        prompt = ChatPromptTemplate.from_template("""You are a financial analyst assistant. Answer questions by synthesizing and interpreting information from the provided documents — do not simply quote or paraphrase them.

Each document section in the context is labeled with [SOURCE: filename] so you know which company it belongs to.

Given the context information and not prior knowledge, answer the following question:
Context: {context}
Question: {question}

Guidelines:
- Lead with your own analytical conclusion, then support it with evidence.
- Summarize and interpret the data; avoid copying sentences directly from the documents.
- When comparing companies, draw a clear comparative insight (e.g. which is higher/lower and why it matters).
- Only mention source filenames briefly in parentheses if needed for clarity, not as the main content.
- If the answer is not in the documents, say "I don't have enough information to answer this question."

You MUST follow this exact formatting structure:

### Section Title

- bullet point one
- bullet point two
- bullet point three

### Next Section Title

- bullet point one
- bullet point two

Rules:
- Every section title must be on its own line, starting with ###.
- Never place ### in the middle of a sentence or on the same line as a bullet.
- Every bullet point must be on its own separate line, starting with -.
- Never place a bullet on the same line as a section title.
- Always leave a blank line between a section title and its bullets, and between sections.
- Do not use asterisks (*) as bullet points.
- Do not write long paragraphs.
""")

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.5)

        def format_docs(docs):
            parts = []
            for doc in docs:
                source = doc.metadata.get("source", "unknown")
                filename = os.path.basename(source)
                parts.append(f"[SOURCE: {filename}]\n{doc.page_content}")
            return "\n\n".join(parts)

        # ------------------------------------------------- #
        # Get response from the chatbot with spinner
        with st.spinner("Thinking..."):
            source_documents = multi_doc_retriever(user_input)
            context = format_docs(source_documents)
            chain = prompt | llm | StrOutputParser()
            response_text = chain.invoke({"context": context, "question": user_input})

            # Display retrieved chunks in an expander if source documents are available
            if source_documents:
                with st.expander("View Retrieved Chunks (Context)"):
                    for i, doc in enumerate(source_documents):
                        st.markdown(f"**Chunk {i+1}**")
                        st.markdown(f"**Content:** {doc.page_content}")
                        st.markdown(f"**Source:** Page {doc.metadata.get('page', 'unknown')}")
                        st.markdown("---")
        # Display assistant response
        # with st.chat_message("assistant"):
        #     st.markdown(response_text)

        # Escape dollar signs to prevent Streamlit from rendering them as LaTeX
        display_text = response_text.replace("$", r"\$")

        with st.chat_message("assistant"):
            st.markdown(display_text)

        # Store assistant response in session state
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("Please upload PDF files to begin.")
