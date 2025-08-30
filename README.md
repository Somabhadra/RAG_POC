# RAG MVP

A lightweight **Retrieval-Augmented Generation (RAG)** proof of concept for semantic search and question answering over your own documents.  
Supports both **interactive CLI** and **Streamlit web UI**.

---

## ✨ Features
- **Document Loading**: Reads multiple `.txt` files from a folder  
- **Text Chunking**: Splits files into manageable chunks  
- **Semantic Search**: Retrieves relevant content with embeddings & stores in VDB  
- **Question Answering**: Uses Microsoft’s **Phi-2** model *(can be replaced by other LLMs or Azure OpenAI for enterprise needs)*  
- **Source Attribution**: Shows which files were used for the answer  
- **Interactive CLI & Streamlit UI**

---

## ⚙️ Installation

```bash
# Install dependencies
pip install faiss-cpu
pip install sentence-transformers transformers torch scikit-learn
pip install streamlit

# Create a folder for your documents
mkdir text_files
```

---

## 🚀 Run the App

### Streamlit Web UI
```bash
streamlit run app.py
```
📸 See example screenshot: `streamlit_example.jpg`

### Terminal / CLI
```bash
python rag_t.py
```

---

## 💡 Usage

1. Place your `.txt` files in the `text_files` directory (or specify a custom path).  
2. Start the app (Streamlit or CLI).  
3. Ask questions about your documents.  
4. Type `exit` to quit (CLI mode).

### Example (CLI)
```
Enter txt dir (default 'text_files'): 
Loading models...
Models ready!
Found 3 file(s).
  - LC.txt: 2 chunks
  - ML.txt: 3 chunks
  - DL.txt: 2 chunks
Created 6 embeddings.
rag_mvp ready! Ask Qs (type 'exit' to quit):

Q: What is AI?
A: Artificial Intelligence (AI) refers to the simulation of human intelligence in machines that are programmed to think like humans and mimic their actions...

Src: ML.txt, LC.txt
```

---

## 🔎 How It Works

```
Documents → Chunking → Embeddings → Vector Search (VDB)
                          ↓
        Query → Embedding → Search → Context → LLM → Answer
```

---

## 🔧 Customization

- **Change Model**: Update `model="microsoft/phi-2"` in `__init__`  
- **Chunk Size**: Modify size in `_split()`  
- **Top-k Retrieval**: Adjust `k` in `search()`  

---

## 🌐 Streamlit App (`app.py`)

To run the web app:
```bash
streamlit run app.py
```

---

## 📦 Requirements
- Python **3.10**  
- `sentence-transformers`  
- `transformers`  
- `torch`  
- `scikit-learn`  
- `numpy`  
- `faiss-cpu`

---

## ⚠️ Limitations
- Works only with `.txt` files  
- No embedding persistence (fresh embeddings created each run)  
- Internet required for first model download  
