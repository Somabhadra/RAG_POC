import streamlit as st
from rag_mvp import rag_mvp

st.set_page_config(page_title="RAG MVP", layout="centered")
st.title("RAG MVP")

# Initialize session state
if "rag" not in st.session_state:
    st.session_state.rag = None

# File upload
uploaded_files = st.file_uploader(
    "Upload text files", 
    type=["txt"], 
    accept_multiple_files=True
)

if uploaded_files and st.button("Process Files"):
    with st.spinner("Processing..."):
        rag = rag_mvp()
        if rag.load_from_texts(uploaded_files) and rag.embed():
            st.session_state.rag = rag
            st.success("Ready to answer questions!")

# Question input
if st.session_state.rag:
    question = st.text_input("Ask a question:")
    
    if question and st.button("Get Answer"):
        with st.spinner("Thinking..."):
            answer, sources = st.session_state.rag.ask(question)
            
            st.write("**A:**")
            st.write(answer)
            
            if sources:
                st.write("**Src:**")
                for source in sources:
                    st.write(f"- {source}")
            
            # Opt: show chunks on expand
            with st.expander("See relevant content"):
                chunks = st.session_state.rag.search(question)
                for chunk in chunks:
                    st.write(f"From {chunk['src']}:")
                    st.write(chunk['txt'][:300] + "..." if len(chunk['txt']) > 300 else chunk['txt'])
                    st.divider()