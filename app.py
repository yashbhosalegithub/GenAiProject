import streamlit as st
from utils import load_qa_chain

st.set_page_config(page_title="RAG Q&A System", layout="wide")

st.title("📄 RAG-based Question Answering System")
st.write("Ask questions from your uploaded documents")

qa_chain = load_qa_chain()

question = st.text_input("❓ Enter your question")

if st.button("Get Answer"):
    if question:
        with st.spinner("Thinking..."):
            answer = qa_chain.run(question)
            st.success("✅ Answer:")
            st.write(answer)
    else:
        st.warning("Please enter a question")
