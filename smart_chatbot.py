import os
from apikey import apikey
import streamlit as st
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = apikey

# App framework
st.title("🦜🔗 Smart Health Chatbot")
prompt = st.text_input("Hi, welcome to our clinic! Please enter your symptoms or questions here")

# Initialize memory in session state if it doesn't exist
if "memory" not in st.session_state:
    st.session_state.memory = []

# Read training handbook from a txt file
file_path = "training_handbook.txt"
with open(file_path, "r") as file:
    training_handbook = file.read()

# Prompt templates
chatbot_template = PromptTemplate(
    input_variables=["topic", "training_handbook"],
    template="Given the training handbook: {training_handbook}\nWhich department should I visit for {topic}?"
)

# LLMs
llm = OpenAI(temperature=0.8)

# Combine LLMChain with Memory
chatbot_chain = LLMChain(llm=llm, prompt=chatbot_template, verbose=True)

def generate_response(topic):
    # Generate response with training handbook as context
    response = chatbot_chain.run(topic=topic, training_handbook=training_handbook)
    
    # Add the user's input to memory
    st.session_state.memory.append(f"<p><strong>You:</strong> {topic}</p>")
    
    # Add the chatbot's response to memory
    st.session_state.memory.append(f"<p><strong>Smart Chatbot:</strong> {response}</p>")

    return response

# Show answers to the screen if there is a prompt
if prompt:
    response = generate_response(prompt)
    st.write(response)

    # Display chat history with tags
    with st.expander("Chat History"):
        for line in st.session_state.memory:
            st.write(line, unsafe_allow_html=True)
