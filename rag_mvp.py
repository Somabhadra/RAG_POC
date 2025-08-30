import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss


# rag_mvp
class rag_mvp:
    def __init__(self, path=None):
        """
        Initialize rag_mvp system with embeddings + LLM with FAISS support.
        - path: directory containing .txt docs (optional if using Streamlit upload)
        """
        self.path = path       # path to txt files
        self.docs = []         # store text chunks
        self.embs = None       # embeddings for chunks
        self.index = None      # FAISS index
        self.emb_dim = 384     # Dimension for all-MiniLM-L6-v2 embeddings

        # Load embedding & generation models
        print("Loading models...")
        self.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.gen_model = pipeline(
            "text-generation",
            model="microsoft/phi-2",  # can replace with another model
            max_new_tokens=150,
            temperature=0.5  # control creativity
        )
        print("Models ready!")

    def load_from_path(self):
        """
        Load .txt files from folder and split into chunks.
        Returns True if successful.
        """
        if not self.path or not os.path.exists(self.path):
            print(f"Path '{self.path}' not found.")
            return False

        files = [f for f in os.listdir(self.path) if f.endswith('.txt')]
        if not files:
            print(f"No txt files in '{self.path}'.")
            return False

        print(f"Found {len(files)} file(s).")

        for fn in files:
            fpath = os.path.join(self.path, fn)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    txt = f.read()

                chunks = self._split(txt)
                for i, c in enumerate(chunks):
                    self.docs.append({"txt": c.strip(), "src": fn, "cid": i})

                print(f"  - {fn}: {len(chunks)} chunks")

            except Exception as e:
                print(f"Error {fn}: {e}")

        return len(self.docs) > 0

    def load_from_texts(self, uploaded_files):
        """
        Load and split docs from uploaded files.
        """
        for file in uploaded_files:
            try:
                text = file.read().decode("utf-8")
                chunks = self._split(text)
                for i, c in enumerate(chunks):
                    self.docs.append({"txt": c.strip(), "src": file.name, "cid": i})
            except Exception as e:
                print(f"Error {file.name}: {e}")

        return len(self.docs) > 0

    def _split(self, txt, size=400):
        """
        Split text into chunks of ~size characters.
        """
        words = txt.split()
        res, buf, buf_len = [], [], 0
        for w in words:
            if buf_len + len(w) > size and buf:
                res.append(" ".join(buf))
                buf, buf_len = [], 0
            buf.append(w)
            buf_len += len(w) + 1
        if buf:
            res.append(" ".join(buf))
        return res

    def embed(self):
        """
        Create embeddings for all loaded chunks and build Vdb.
        """
        if not self.docs:
            print("No docs to embed.")
            return False
        
        txts = [d["txt"] for d in self.docs]
        
        # Create embeddings
        embeddings = self.emb_model.encode(txts)
        self.embs = embeddings  
        
        # Convert to numpy array for vdb
        embeddings = np.array(embeddings).astype('float32')
        
        # Create vdb
        self.index = faiss.IndexFlatL2(self.emb_dim)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        print(f"Created vdb with {len(self.docs)} embeddings.")
        return True

    def search(self, q, k=3):
        """
        Retrieve top-k most relevant chunks using vdb.
        """
        if self.index is None:
            print("Run embed() first.")
            return []
        
        # Embed query
        q_emb = self.emb_model.encode([q])
        q_emb = np.array(q_emb).astype('float32')
        
        # Search in faiss index
        distances, indices = self.index.search(q_emb, k)
        
        # Get the relevant documents
        results = []
        for idx in indices[0]:
            if idx < len(self.docs):  # Ensure index is valid
                results.append(self.docs[idx])
        
        return results

    def gen(self, q, chunks):
        """
        Generate answer.
        """
        ctx = "\n".join([f"- {c['txt']}" for c in chunks])
        prompt = f"""Use the context to answer. If not sure, say so.

Context:
{ctx}

Q: {q}
A:"""
        try:
            out = self.gen_model(prompt, return_full_text=False)
            return out[0]['generated_text'].strip()
        except:
            return chunks[0]['txt'][:200] + "..." if chunks else "No info."

    def ask(self, q, k=3):
        """
        Full rag_mvp pipeline: retrieve + generate.
        """
        chunks = self.search(q, k=k)
        if not chunks:
            return "No relevant info.", []
        ans = self.gen(q, chunks)
        srcs = list(set([c['src'] for c in chunks]))
        return ans, srcs

    # FAISS-specific methods
    def save_index(self, filepath="faiss_index.bin"):
        """Save FAISS index to file"""
        if self.index:
            faiss.write_index(self.index, filepath)
            print(f"Index saved to {filepath}")
            return True
        return False

    def load_index(self, filepath="faiss_index.bin"):
        """Load FAISS index from file"""
        if os.path.exists(filepath):
            self.index = faiss.read_index(filepath)
            print(f"Index loaded from {filepath}")
            return True
        else:
            print(f"Index file {filepath} not found")
            return False

    # Backward compatibility methods
    def load(self):
        """Alias for load_from_path for backward compatibility"""
        return self.load_from_path()