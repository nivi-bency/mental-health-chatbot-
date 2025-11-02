import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from tqdm import tqdm

# Load environment variables
load_dotenv()
CHROMA_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")


def load_pdf(path):
    print(f"📘 Loading PDF: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File not found at: {path}")
    loader = PyPDFLoader(path)
    docs = loader.load()
    print(f"✅ Loaded {len(docs)} pages.")
    return docs


def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    print("✂️ Splitting documents into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    split_docs = splitter.split_documents(docs)
    print(f"✅ Created {len(split_docs)} chunks.")
    return split_docs


def create_embeddings():
    print("🧠 Loading SentenceTransformer embeddings (all-MiniLM-L6-v2)...")
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_chroma(docs, embeddings, persist_directory=CHROMA_DIR):
    print("💾 Building Chroma vector database...")
    vectordb = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    print("✅ Chroma DB saved to:", persist_directory)
    return vectordb


if __name__ == "__main__":
    # ✅ Explicit absolute path to your file
    pdf_path = "/Users/nivetha/Desktop/mental-health-chatbot/data/mental_health_Document.pdf"

    docs = load_pdf(pdf_path)
    chunks = split_documents(docs)
    embeddings = create_embeddings()
    build_chroma(chunks, embeddings)

    print("🎉 Database setup complete!")
