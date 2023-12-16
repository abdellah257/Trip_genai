import streamlit as st
import requests

from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.prompts import StringPromptTemplate
from langchain.utilities import SerpAPIWrapper
from langchain.chains import LLMChain
from langchain.llms import LlamaCpp
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import re
from llama_cpp import Llama


callback_manager = AsyncCallbackManager([StreamingStdOutCallbackHandler()])


n_gpu_layers = 1  # Metal set to 1 is enough.
n_batch = 512  # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="./llama-2-7b.Q4_K_M.gguf",
    n_gpu_layers=n_gpu_layers,
    n_batch=n_batch,
    max_tokens=500,
    top_p=0.95,
    frequency_penalty=0.5,
    presence_penalty=0.5,
    temperature = 0.3,
    f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
    grammar_path="gr.gbnf",
)

prompt = """
You are an intelligent travel assistant. Provide a detailed travel plan to Marrakech, including a section for each day with a detailed outline of the schedule. Include timestamps, locations, and contact details where necessary.
"""


llm(prompt)

#headers = {"Authorization": f"Bearer {'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'}"}


#API_URL = "https://api-inference.huggingface.co/models/gpt2"


#def query(payload):
#    response = requests.post(API_URL, headers=headers, json=payload)
#    return response.json()

#[theme]
#base="light"
#backgroundColor="#e2d2bd"
#secondaryBackgroundColor="#d8caab"

# Function to generate the itinerary based on user input or default value
def fctgai(user_input='default'):
    if user_input == '':
        return "L'itinéraire est comme suit : Arrivée Casablanca -> Fes -> Oujda -> Nador -> retour"
    else:

        # Make a request to the Hugging Face Inference API
        response = llm(prompt+user_input )   
        
        # Customize the logic based on your requirements
        return f"L'itinéraire est comme suit : {response} "

# Set Streamlit app title and description
st.title('Itinerary Planning App')
st.image('logo.png', caption='', use_column_width=True)
st.write("The best app for itinerary planning in Morocco")

# Set theme colors
st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Input text zone
user_input = st.text_area('Enter your itinerary text here:', '')

# Button to validate the text and generate itinerary
if st.button('Validate Itinerary'):
    result = fctgai(user_input)
    st.success(result)

# Additional content or actions can be added here

# Footer or additional information
st.markdown('© 2023 RAM-DT team hackathon - MoroccoAI')
# Run the app with: streamlit run your_script_name.py
