import os
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

# rag_mvp with FAISS
class rag_mvp:
    def __init__(self, path="<path>"):
        """
        Initialize rag_mvp system with embeddings + LLM using FAISS.
        - path: directory containing .txt docs
        """
        self.path = path       # path to txt files
        self.docs = []         # store text chunks
        self.index = None      # FAISS index
        self.emb_dim = 384     # Dimension for all-MiniLM-L6-v2 embeddings

        # Load embedding + generation models
        print("Loading models...")
        self.emb_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.gen_model = pipeline(
            "text-generation",
            model="microsoft/phi-2",
            max_new_tokens=150,
            temperature=0.5
        )
        print("Models ready!")

    def load(self):
        """
        Load .txt files from folder and split into chunks.
        Returns True if successful.
        """
        if not os.path.exists(self.path):
            print(f"Path '{self.path}' not found.")
            return False

        files = [f for f in os.listdir(self.path) if f.endswith('.txt')]
        if not files:
            print(f"No txt files in '{self.path}'.")
            return False

        print(f"Found {len(files)} file(s).")

        # Process each file
        for fn in files:
            fpath = os.path.join(self.path, fn)
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    txt = f.read()

                # Split into chunks
                chunks = self._split(txt)
                for i, c in enumerate(chunks):
                    self.docs.append({
                        "txt": c.strip(),  # text content
                        "src": fn,         # source filename
                        "cid": i           # chunk id
                    })

                print(f"  - {fn}: {len(chunks)} chunks")

            except Exception as e:
                print(f"Error {fn}: {e}")

        return len(self.docs) > 0

    def _split(self, txt, size=400):
        """
        Split text into chunks of ~size characters.
        - Returns list of text chunks
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
        Create embeddings for all loaded chunks and build FAISS index.
        Returns True if successful.
        """
        if not self.docs:
            print("No docs to embed.")
            return False

        txts = [d["txt"] for d in self.docs]
        embeddings = self.emb_model.encode(txts)
        
        # Convert to numpy array
        embeddings = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        self.index = faiss.IndexFlatL2(self.emb_dim)
        
        # Add embeddings to index
        self.index.add(embeddings)
        
        print(f"Created FAISS index with {len(self.docs)} embeddings.")
        return True

    def search(self, q, k=3):
        """
        Retrieve top-k most relevant chunks for a query using FAISS.
        - q: query text
        - k: number of results
        """
        if self.index is None:
            print("Run embed() first.")
            return []

        # Embed query
        q_emb = self.emb_model.encode([q])
        q_emb = np.array(q_emb).astype('float32')
        
        # Search in FAISS index
        distances, indices = self.index.search(q_emb, k)
        
        # Get the relevant documents
        results = []
        for idx in indices[0]:
            if idx < len(self.docs):  # Ensure index is valid
                results.append(self.docs[idx])
        
        return results

    def gen(self, q, chunks):
        """
        Generate answer using retrieved chunks + LLM.
        - q: user query
        - chunks: retrieved context
        """
        # Build context string
        ctx = "\n".join([f"- {c['txt']}" for c in chunks])

        # Prompt for LLM
        prompt = f"""Use the context to answer. If not sure, say so.

Context:
{ctx}

Q: {q}
A:"""

        try:
            out = self.gen_model(prompt, return_full_text=False)
            return out[0]['generated_text'].strip()
        except:
            # Fallback → return raw chunk
            return chunks[0]['txt'][:200] + "..." if chunks else "No info."

    def ask(self, q):
        """
        Full rag_mvp pipeline for a query:
        - Retrieve chunks using FAISS
        - Generate answer
        Returns (answer, sources)
        """
        chunks = self.search(q)
        if not chunks:
            return "No relevant info.", []

        ans = self.gen(q, chunks)
        srcs = list(set([c['src'] for c in chunks]))
        return ans, srcs

    def save_index(self, filepath="faiss_index.bin"):
        """Save FAISS index to file"""
        if self.index:
            faiss.write_index(self.index, filepath)
            print(f"Index saved to {filepath}")

    def load_index(self, filepath="faiss_index.bin"):
        """Load FAISS index from file"""
        if os.path.exists(filepath):
            self.index = faiss.read_index(filepath)
            print(f"Index loaded from {filepath}")
            return True
        else:
            print(f"Index file {filepath} not found")
            return False


def main():
    """
    Run interactive rag_mvp pipeline.
    """
    # Ask user for input path
    path = input("Enter txt dir (default 'text_files'): ").strip() or "text_files"

    rag = rag_mvp(path)

    # load existing index
    index_file = "faiss_index.bin"
    if rag.load_index(index_file):
        if not rag.load():
            return
    else:
        # Load + embed documents and create new index
        if not rag.load():
            return
        if not rag.embed():
            return
        # Save the index
        rag.save_index(index_file)

    print("\nrag_mvp ready! Ask Qs (type 'exit' to quit):")

    # Continuous RAG MVP
    while True:
        try:
            q = input("\nQ: ").strip()
            if q.lower() in ['exit', 'quit']:
                break
            if not q:
                continue

            ans, srcs = rag.ask(q)

            print(f"\nA: {ans}")
            if srcs:
                print(f"Src: {', '.join(srcs)}")

        except KeyboardInterrupt:
            print("\nBye.")
            break
        except Exception as e:
            print(f"Err: {e}")


if __name__ == "__main__":
    main()