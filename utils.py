
import os

import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

VECTORSTORE_PATH = "vectorstore"


def get_groq_api_key():
    # Read the key from Streamlit Cloud Secrets
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return key
    except Exception:
        pass

    # Read the key from local .env
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key

    raise RuntimeError(
        "GROQ_API_KEY is missing. Add it to Streamlit Secrets "
        "or your local .env file."
    )


def load_qa_chain():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True,
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    llm = ChatGroq(
        model="openai/gpt-oss-20b",
        temperature=0,
        api_key=get_groq_api_key(),
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
    )

    return qa_chain
