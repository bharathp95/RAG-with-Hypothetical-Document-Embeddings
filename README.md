\# Chat with your PDF using HyDE — Full Code Explanation

A RAG pipeline that uses **Hypothetical Document Embeddings (HyDE)** to retrieve more relevant chunks from a PDF and answer questions using an LLM.

---

## What is HyDE?

Normal RAG embeds the user's question and searches for similar chunks. The problem is questions and answers sound very different:

```
question: "what are the side effects?"
chunk:    "patients reported nausea, headache and fatigue"
```

These two have low similarity even though the chunk is exactly what you need.

HyDE fixes this by asking the LLM to **generate a hypothetical answer** first, then embedding that instead:

```
Normal RAG:   question → embed → search
HyDE:         question → LLM generates fake answer → embed fake answer → search
```

A fake answer sounds like a real answer — so it matches the chunks much better.

---

## The Full Pipeline

```
PDF uploaded
    ↓
extract text (PyMuPDF)
    ↓
split into overlapping chunks
    ↓
embed each chunk (SentenceTransformer)
    ↓
store in ChromaDB
    ↓
user asks a question
    ↓
LLM generates a hypothetical answer to the question
    ↓
embed the hypothetical answer
    ↓
search ChromaDB with that embedding → top 3 relevant chunks
    ↓
send real chunks + question to LLM
    ↓
final answer displayed on Streamlit
```

---

## Libraries Used

| Library | Purpose |
|---|---|
| `streamlit` | Frontend UI |
| `fitz` (PyMuPDF) | Extract text from PDF |
| `chromadb` | Vector database to store and search embeddings |
| `sentence-transformers` | Convert text into embeddings (384 numbers) |
| `groq` | Call the LLM API |

---

## Code Breakdown

### 1. Background Gradient

```python
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #1a1a2e, #16213e, #0f3460);
    }
    </style>
""", unsafe_allow_html=True)
```

Injects raw CSS into the Streamlit app to style the background with a dark blue navy gradient. `unsafe_allow_html=True` is required to allow raw HTML and CSS in Streamlit.

---

### 2. Setup

```python
model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="mps",
    local_files_only=True
)
```

Loads the embedding model. Three things:

- `all-MiniLM-L6-v2` — lightweight pre-trained model that converts text into 384 numbers (embeddings)
- `device="mps"` — uses Apple Silicon GPU instead of CPU, much faster
- `local_files_only=True` — uses already downloaded model, no internet needed

```python
client = chromadb.PersistentClient(path="hyde_db")
```

Opens or creates a local vector database saved to disk in a folder called `hyde_db`. Persistent means it survives after the program closes.

```python
groq_client = Groq(api_key="your_api_key")
```

Initializes the Groq client to call the LLM.

---

### 3. extract_text()

```python
def extract_text(pdf_file):
    doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text
```

- `fitz.open(stream=...)` — opens PDF from memory. Streamlit uploads files as streams not file paths so you use `stream=` instead of a path
- `for page in doc` — loops through every page of the PDF
- `page.get_text()` — extracts raw text from that page
- Returns one big string of all text from the entire PDF

---

### 4. split_chunks()

```python
def split_chunks(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks
```

You can't embed the entire PDF at once — it's too large. So you split it into smaller pieces called chunks.

- `chunk_size=500` — each chunk is 500 characters long
- `overlap=50` — each chunk shares 50 characters with the previous chunk

Why overlap? To avoid losing context at the edges:

```
chunk 1: "...the function returns a"
chunk 2: "a value based on the input..."  ← overlap preserves this connection
```

- `start = end - overlap` — steps back 50 characters before starting the next chunk, creating the overlap

---

### 5. store_chunks()

```python
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
```

- `client.list_collections()` — gets all existing collections
- `if "pdf_chunks" in existing` — checks before deleting so it doesn't crash on first run when collection doesn't exist yet
- `client.delete_collection(...)` — wipes old PDF chunks so they don't mix with the new PDF
- `enumerate(chunks)` — gives both the index `i` and chunk text so you can create unique ids
- `model.encode(chunk).tolist()` — converts chunk text into 384 numbers. `.tolist()` converts from numpy array to plain Python list because ChromaDB requires that
- `collection.add(...)` — stores three things per chunk:
  - `ids` — unique label like `chunk_0`, `chunk_1`
  - `embeddings` — the 384 numbers used for searching
  - `documents` — the original text returned after a match is found

---

### 6. generate_hypothetical_answer() — The HyDE Step

```python
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
```

This is the core of HyDE. Instead of searching with the question directly, you ask the LLM to imagine what an answer looks like. That imagined answer will use vocabulary and phrasing that is much closer to what's actually in your PDF chunks.

Example:
```
question:            "what are the side effects?"
hypothetical answer: "the side effects include nausea, headache and fatigue"
                              ↑
                    now this matches your chunks
```

---

### 7. retrieve_with_hyde()

```python
def retrieve_with_hyde(question, collection, top_k=3):
    hypothetical = generate_hypothetical_answer(question)
    hyde_embedding = model.encode(hypothetical).tolist()
    results = collection.query(
        query_embeddings=[hyde_embedding],
        n_results=top_k
    )
    return results["documents"][0]
```

- `generate_hypothetical_answer(question)` — gets the fake answer from the LLM
- `model.encode(hypothetical)` — embeds the fake answer, not the original question
- `collection.query(...)` — ChromaDB compares the hypothetical embedding against every stored chunk embedding using cosine similarity and returns the closest matches
- `n_results=3` — return top 3 most relevant chunks
- `results["documents"][0]` — the `[0]` removes the outer list ChromaDB wraps results in

---

### 8. ask()

```python
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
```

- `retrieve_with_hyde(...)` — gets the top 3 real chunks using HyDE
- `"\n\n".join(chunks)` — joins the 3 chunks into one string with blank lines between them
- System prompt tells the LLM to only answer from provided context — prevents hallucination
- User message contains both the context (retrieved chunks) and the original question
- `model="openai/gpt-oss-120b"` — OpenAI's open weight 120B Mixture of Experts model running on Groq at 500+ tokens per second

---

### 9. Streamlit UI

```python
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
```

- `st.file_uploader(...)` — renders a file upload button, only accepts PDFs
- `if uploaded_file:` — everything inside only runs after a file is uploaded
- `st.spinner(...)` — shows a loading animation while processing
- `st.success(...)` — shows a green success message with chunk count
- `st.text_input(...)` — renders a text box for the user's question
- `if question:` — only runs after the user types something and hits enter
- `st.write(answer)` — displays the LLM's final answer

---

## HyDE vs Normal RAG — Side by Side

| | Normal RAG | HyDE |
|---|---|---|
| What gets embedded | The question | A hypothetical answer |
| Search quality | Good | Better for vague questions |
| LLM calls per query | 1 | 2 (one for hypothesis, one for answer) |
| Speed | Faster | Slightly slower |
| Best for | Specific questions | Vague or exploratory questions |

---

## Key Concepts Summary

| Concept | What it means |
|---|---|
| Embedding | A list of 384 numbers representing the meaning of text |
| Vector Database | A database that stores and searches embeddings |
| Cosine Similarity | Measures how similar two embeddings are (0 to 1) |
| Chunking | Splitting large text into smaller pieces for embedding |
| Overlap | Shared characters between chunks to preserve context |
| HyDE | Generate a fake answer → embed it → search → get better results |

---

