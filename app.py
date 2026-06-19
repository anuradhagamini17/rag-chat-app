import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb
import google.generativeai as genai

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📚",
    layout="wide"
)

st.title("📚 PDF RAG Chatbot")

# ----------------------------
# Gemini API Setup
# ----------------------------
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# ----------------------------
# Chat History
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ----------------------------
# Load Embedding Model
# ----------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embedding_model = load_model()

# ----------------------------
# ChromaDB
# ----------------------------
@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection("pdf_collection")
    except:
        pass

    return client.get_or_create_collection(
        name="pdf_collection"
    )

collection = get_collection()

# ----------------------------
# Upload PDF
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file:

    # Read PDF
    pdf = PdfReader(uploaded_file)

    text = ""

    for page in pdf.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text

    # Empty PDF Check
    if not text.strip():
        st.error("No text could be extracted from this PDF.")
        st.stop()

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_text(text)

    # Create embeddings
    embeddings = embedding_model.encode(chunks)

    # Clear previous collection
    try:
        collection.delete(
            ids=[str(i) for i in range(100000)]
        )
    except:
        pass

    ids = [str(i) for i in range(len(chunks))]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings.tolist()
    )

    st.success(f"Created {len(chunks)} chunks")
    st.success(f"Created {len(embeddings)} embeddings")
    st.success("Stored embeddings in ChromaDB")

    # Preview chunks
    with st.expander("View First 3 Chunks"):
        for i, chunk in enumerate(chunks[:3]):
            st.markdown(f"### Chunk {i+1}")
            st.write(chunk)

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # User question
    question = st.chat_input(
        "Ask a question about the PDF..."
    )

    if question:

        # Store user message
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Create query embedding
        question_embedding = embedding_model.encode(
            [question]
        )

        # Retrieve relevant chunks
        results = collection.query(
            query_embeddings=question_embedding.tolist(),
            n_results=3
        )

        retrieved_chunks = results["documents"][0]

        context = "\n\n".join(retrieved_chunks)

        prompt = f"""
You are a helpful AI assistant.

Answer ONLY using the information provided in the context.

If the answer is not available in the context, reply:

"I could not find that information in the document."

Context:
{context}

Question:
{question}
"""

        with st.spinner("Generating answer..."):

            gemini_model = genai.GenerativeModel(
                "gemini-1.5-flash"
            )

            response = gemini_model.generate_content(
                prompt
            )

            answer = response.text

        # Store assistant message
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        with st.chat_message("assistant"):
            st.markdown(answer)

        # Show sources
        with st.expander("📄 Retrieved Source Chunks"):
            for i, chunk in enumerate(retrieved_chunks):
                st.markdown(f"### Source {i+1}")
                st.write(chunk)