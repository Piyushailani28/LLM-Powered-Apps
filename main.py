'''import time
import streamlit as st
import os
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt_template = ChatPromptTemplate.from_template( 
    """
    You are a knowledgeable and helpful assistant.
    Use ONLY the context provided below to answer the question. Do NOT use prior knowledge or make up answers. If the answer is not contained in the context, respond with "I'm sorry, the answer is not available in the provided context."

    <context>
    {context}
    </context>

    Question: {input}

    Answer in a clear, concise, and accurate manner.
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("Research Paper Q&A")

user_question = st.text_input("Enter the questions from the research paper")  

if st.button("Document Embedding"):
    create_vector_embedding()
    st.write("Vector Database Ready")

# check if vector store exists before running retrieval
if user_question:
    if "vectors" not in st.session_state:
        st.warning("Please create document embeddings first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt_template)  # changed from `prompt` to `prompt_template`
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({"input": user_question})  # changed from `prompt` to `user_question`
        print(f"Response time : {time.process_time() - start}")

        st.write(response["answer"])

        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response['context']):
                st.write(doc.page_content)
                st.write("--------------------------------")
'''


import time
import streamlit as st
import os
from tempfile import NamedTemporaryFile
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory  # Added for memory
from dotenv import load_dotenv

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-8b-8192")

prompt_template = ChatPromptTemplate.from_template(
    """
    You are a knowledgeable and helpful assistant.
    Use ONLY the context provided below and the chat history to answer the question.
    If the answer is not contained in the context, respond with "I'm sorry, the answer is not available in the provided context."

    <context>
    {context}
    </context>

    Chat History:
    {chat_history}

    Question: {input}

    Answer in a clear, concise, and accurate manner.
    """
)

st.set_page_config(page_title="PDF Chat Assistant")
st.title("📄 Chat with Your PDF")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True, input_key="input", memory_key="chat_history")  # Added memory setup

uploaded_file = st.file_uploader("Upload a PDF to begin", type=["pdf"])

if uploaded_file:
    with st.spinner("Processing your PDF..."):
        with NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        st.session_state.retriever = vectorstore.as_retriever()

    st.success("PDF processed. You can now ask questions!")

user_input = st.chat_input("Ask something about the uploaded PDF")

if user_input:
    if "retriever" not in st.session_state:
        st.warning("Please upload a PDF first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt_template)
        retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)

        with st.spinner("Generating answer..."):
            response = retrieval_chain.invoke({
                "input": user_input,
                "chat_history": st.session_state.memory.buffer  # Added memory injection
            })

            st.session_state.memory.chat_memory.add_user_message(user_input)  # Add user message to memory
            st.session_state.memory.chat_memory.add_ai_message(response["answer"])  # Add bot response to memory

            st.session_state.chat_history.append(("user", user_input))
            st.session_state.chat_history.append(("assistant", response["answer"]))

for role, message in st.session_state.chat_history:
    if role == "user":
        st.chat_message("user").write(message)
    else:
        st.chat_message("assistant").write(message)
