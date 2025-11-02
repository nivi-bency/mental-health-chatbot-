import streamlit as st
import os
from dotenv import load_dotenv

# ✅ Correct imports for LangChain v1.0+
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ======================================
# 🔧 Load environment variables
# ======================================
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DB_DIR = os.getenv("CHROMA_DB_DIR", "./chroma_db")

# ======================================
# ⚙️ Streamlit UI configuration
# ======================================
st.set_page_config(page_title="🧠 MindEase Chatbot", layout="wide")
st.title("🩵 MindEase: Your Mental Health Assistant")
st.markdown("This chatbot uses **Groq + LangChain + Chroma** to answer mental health–related queries with empathy and professionalism.")

# ======================================
# 🧠 Load Vector Database
# ======================================
@st.cache_resource
def load_vectorstore():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
    return vectordb

vectordb = load_vectorstore()
retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# ======================================
# 🤖 Initialize LLM (Groq)
# ======================================
@st.cache_resource
def load_llm():
    return ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")



llm = load_llm()

# ======================================
# 🧩 Define Custom Retrieval Function
# ======================================
def retrieve_and_answer(query):
    # ✅ New retriever call — works for your LangChain version
    docs = retriever.invoke(query)  # instead of get_relevant_documents()
    if not docs:
        return "I'm here to help, but I couldn't find relevant info. Could you rephrase that?"

    context = "\n\n".join([d.page_content for d in docs])

    prompt = ChatPromptTemplate.from_template("""
You are a compassionate and well-informed mental health assistant.
Use the context below to provide a calm, supportive, and helpful response.
If you are unsure, gently suggest consulting a mental health professional.

<context>
{context}
</context>

Question: {question}
""")

    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"context": context, "question": query})
    return response


# ======================================
# 💬 Chat Interface
# ======================================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_query = st.text_input("💬 Ask about mental health, emotions, or coping strategies:")

if st.button("Send") and user_query:
    with st.spinner("Thinking..."):
        answer = retrieve_and_answer(user_query)
        st.session_state.chat_history.append(("🧍‍♀️ You", user_query))
        st.session_state.chat_history.append(("🤖 MindEase", answer))

# Display chat history
if st.session_state.chat_history:
    st.markdown("### 🗨️ Conversation")
    for role, text in st.session_state.chat_history:
        st.markdown(f"**{role}:** {text}")
