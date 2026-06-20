import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from groq import Groq
import chromadb
import os


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
# Groq Setup
# ----------------------------
client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)


# ----------------------------
# Chat Memory
# ----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------
# Embedding Model
# ----------------------------
@st.cache_resource
def load_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


embedding_model = load_model()



# ----------------------------
# ChromaDB
# ----------------------------
@st.cache_resource
def get_collection():

    os.makedirs(
        "chroma_db",
        exist_ok=True
    )

    db = chromadb.PersistentClient(
        path="chroma_db"
    )

    return db.get_or_create_collection(
        name="pdf_collection"
    )


collection = get_collection()



# ----------------------------
# Upload PDF
# ----------------------------
uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)



if uploaded_file:


    reader = PdfReader(
        uploaded_file
    )


    text = ""


    for page in reader.pages:

        content = page.extract_text()

        if content:
            text += content



    if not text.strip():

        st.error(
            "No readable text found in PDF"
        )

        st.stop()



    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )


    chunks = splitter.split_text(
        text
    )



    embeddings = embedding_model.encode(
        chunks
    )



    # clear old data
    try:

        old = collection.get()

        if old["ids"]:

            collection.delete(
                ids=old["ids"]
            )

    except Exception:
        pass



    collection.add(

        ids=[
            str(i)
            for i in range(len(chunks))
        ],

        documents=chunks,

        embeddings=embeddings.tolist()
    )



    st.success(
        f"Stored {len(chunks)} chunks"
    )



# ----------------------------
# Show Chat History
# ----------------------------
for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )



# ----------------------------
# Chat Input
# ----------------------------
question = st.chat_input(
    "Ask something about your PDF..."
)



if question:


    st.session_state.messages.append(
        {
            "role":"user",
            "content":question
        }
    )


    with st.chat_message("user"):

        st.markdown(question)



    query_embedding = embedding_model.encode(
        [question]
    )



    results = collection.query(

        query_embeddings=
        query_embedding.tolist(),

        n_results=3
    )



    context = "\n\n".join(
        results["documents"][0]
    )



    prompt = f"""

You are a PDF assistant.

Answer only using the context.

If the answer is not available say:

"I could not find that information in the document."

Context:

{context}


Question:

{question}

"""



    with st.spinner(
        "Thinking..."
    ):


        try:

            response = client.chat.completions.create(

                model="llama-3.3-70b-versatile",

                messages=[
                    {
                        "role":"user",
                        "content":prompt
                    }
                ]

            )


            answer = response.choices[0].message.content



        except Exception as e:

            answer = (
                "Groq API error. "
                "Check your API key or quota."
            )



    st.session_state.messages.append(

        {
            "role":"assistant",
            "content":answer
        }

    )



    with st.chat_message("assistant"):

        st.markdown(answer)



    with st.expander(
        "📄 Sources"
    ):

        for i, doc in enumerate(
            results["documents"][0]
        ):

            st.markdown(
                f"### Source {i+1}"
            )

            st.write(doc)