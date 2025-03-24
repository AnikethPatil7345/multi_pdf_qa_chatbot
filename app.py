import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain 
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ Google API Key not found! Please add it to the .env file.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# Function to extract text from PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            extracted_text = page.extract_text()
            if extracted_text:
                text += extracted_text + "\n"
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Function to store text chunks as vectors in FAISS
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to create a conversational Q&A chain
def get_conversational_chain():
    prompt_template = """
    Answer the questions as detailed as possible from the provided context. 
    If the answer is not available in the provided context, just say:
    "Answer is not available in the context." 
    Do not provide incorrect answers.

    Context:
    {context}

    Question:
    {question}

    Answer:"""

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")  # Use latest Gemini model

    return load_qa_chain(llm, chain_type="stuff", prompt=prompt)

# Function to handle user queries
def user_input(user_query):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Load FAISS vector store
    try:
        vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"❌ Error loading vector database: {e}")
        return

    # Search for relevant document chunks
    docs = vector_store.similarity_search(user_query, k=3)

    # Load conversational chain
    chain = get_conversational_chain()

    # Generate response
    response = chain.run(input_documents=docs, question=user_query)

    # Display response
    st.write("**Reply:**", response)

# Main Streamlit App
def main():
    st.title("📄 AI-Powered PDF Q&A Chatbot")

    st.sidebar.header("Upload PDFs")
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", accept_multiple_files=True, type=["pdf"])

    if uploaded_files:
        st.sidebar.success("✅ PDFs uploaded successfully! Processing...")

        # Extract text from uploaded PDFs
        text = get_pdf_text(uploaded_files)

        # Split text into chunks
        text_chunks = get_text_chunks(text)

        # Generate and store vector embeddings
        get_vector_store(text_chunks)
        st.sidebar.success("✅ Text processed and stored in vector database!")

    # User Query Section
    user_query = st.text_input("Ask a question about the PDF:")
    if user_query:
        user_input(user_query)

if __name__ == "__main__":
    main()
