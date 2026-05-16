import streamlit as st
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    }
    </style>
""", unsafe_allow_html=True)

# setup
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="mps",
    local_files_only=True
)
client = chromadb.PersistentClient(path="hyde_db")
groq_client = Groq(api_key="")


def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def split_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


def store_chunks(chunks):
    existing = [c.name for c in client.list_collections()]
    if "pdf_chunks" in existing:
        client.delete_collection("pdf_chunks")
    
    collection = client.get_or_create_collection("pdf_chunks")
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=[embedding],
            documents=[chunk]
        )
    return collection


def generate_hypothetical_answer(question):
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "user",
                "content": f"Write a short paragraph that would answer this question: {question}"
            }
        ]
    )
    return response.choices[0].message.content


def retrieve_with_hyde(question, collection, top_k=3):
    hypothetical = generate_hypothetical_answer(question)
    hyde_embedding = model.encode(hypothetical).tolist()
    results = collection.query(
        query_embeddings=[hyde_embedding],
        n_results=top_k
    )
    return results["documents"][0]


def ask(question, collection):
    chunks = retrieve_with_hyde(question, collection)
    context = "\n\n".join(chunks)
    response = groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Answer using only the context provided."
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {question}"
            }
        ]
    )
    return response.choices[0].message.content


# UI
st.title("Chat with your PDF with Hypothetical Document Embeddings")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    with st.spinner("Indexing your PDF..."):
        text = extract_text(uploaded_file)
        chunks = split_chunks(text)
        collection = store_chunks(chunks)
    st.success(f"Ready! Indexed {len(chunks)} chunks.")

    question = st.text_input("Ask a question about your PDF")

    if question:
        with st.spinner("Generating hypothetical answer and searching..."):
            answer = ask(question, collection)
        st.write(answer)