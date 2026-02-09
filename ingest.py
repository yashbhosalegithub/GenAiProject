import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/documents"
VECTORSTORE_PATH = "vectorstore"

def load_documents():
    documents = []
    for file in os.listdir(DATA_PATH):
        file_path = os.path.join(DATA_PATH, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            continue
        documents.extend(loader.load())
    return documents

def main():
    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(VECTORSTORE_PATH)

    print("✅ Documents indexed successfully using LOCAL embeddings!")

if __name__ == "__main__":
    main()
