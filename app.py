'''import streamlit as st
import openai
#from langchain_openai import ChatOpenAI
from langchain_google_gemini import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
os.environ["GEMINI_API_KEY"] = api_key

## Langsmith Tracking
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Simple Q&A Chatbot" 

## Define the prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "you are an expert phsycologist. you have 10 years of experience and answer questions about mental health"),
        ("user", "Question : {question}")
    ]
)

## Define the function to generate response
def generate_response(question, api_key, llm, temperature, max_tokens):
    llm = ChatGoogleGenerativeAI(model=llm)
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question' : question})
    return answer

## title of the app
st.title("Mental Health Chatbot")

## sidebar for settings
st.sidebar.title("settings")
api_key = st.sidebar.text_input("Enter your Gemini API Key", type="password")

## Drop Down to select various Open AI Models
llm = st.sidebar.selectbox("select an Gemini Model", ["gemini-pro", "gemini-1.5-pro"])

## Slider to adjust temperature
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

## Slider to adjust max tokens
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=1000, value=500, step=100)

## Main panel for chatting
st.write("Go Ahead ask the questions")
user_input = st.text_input("Enter your question here")

if user_input:
    response = generate_response(user_input, api_key, llm, temperature, max_tokens)
    st.write(response)
elif user_input:
    st.write("Please provide the question")

'''

import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
os.environ["GOOGLE_API_KEY"] = api_key

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY', '')
os.environ['LANGCHAIN_TRACING_V2'] = "true"
os.environ['LANGCHAIN_PROJECT'] = "Simple Q&A Chatbot"

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a Gen AI Expert and you have all the answers to the questions whether it is theoritical or practical. You are also a great teacher and you can explain the concepts in a very simple and easy to understand manner."),
        ("user", "Question: {question}")
    ]
)

def generate_response(question, temperature, max_tokens):
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro-latest",
        temperature=temperature,
        max_output_tokens=max_tokens,
        convert_system_message_to_human=True
    )
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    answer = chain.invoke({'question': question})
    return answer

st.title("🧠 GEN AI Teacher")

st.sidebar.title("⚙️ Settings")

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=2048, value=500, step=100)

st.write("Go ahead and ask your questions!")

user_input = st.text_input("Enter your question here:")

if user_input:
    response = generate_response(user_input, temperature, max_tokens)
    st.write("Answer:")
    st.write(response)
else:
    st.write("Please type your question to get started.")