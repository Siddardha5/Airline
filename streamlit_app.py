import os
import streamlit as st
from langchain.llms import OpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableBranch

# Set up the OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

# Initialize the language model
llm = OpenAI(openai_api_key=os.environ["OPENAI_API_KEY"])

# Define the template to determine if the trip is international or national
airline_template = """You are an expert at booking airline tickets.
From the following text, determine whether the flight type is international or national.

Do not respond with more than one word.

Text:
{request}
"""

flight_type_chain = (
    PromptTemplate.from_template(airline_template)
    | llm
    | StrOutputParser()
)

# Define the international chain for visa requirements
international_chain = PromptTemplate.from_template(
    """You are a travel agent that is experienced with immigration and visa requirements. \
Determine the kind of visa the traveller needs for their trip from the following text.
Do not respond with any reasoning. Just respond professionally as a travel agent. Respond in first-person mode.

Your response should follow these guidelines:
    1. Do not provide any reasoning behind the need for visa. Just respond professionally as a travel chat agent.
    2. Address the customer directly

Text:
{text}
"""
) | llm

# Define the general chain for national trips
general_chain = PromptTemplate.from_template(
    """You are a travel agent.
    Given the text below, determine the length of the traveller's journey in hours.

    Your response should follow these guidelines:
    1. You will wish the traveller a safe trip and that they enjoy the next X hour, where X is the length of their flights.
    2. Do not respond with any reasoning. Just respond professionally as a travel chat agent.
    3. Address the customer directly

Text:
{text}
"""
) | llm

# Set up the branching logic
branch = RunnableBranch(
    (lambda x: "international" in x["flight_type"].lower(), international_chain),
    general_chain,
)

# Combine the chains
full_chain = {"flight_type": flight_type_chain, "text": lambda x: x["request"]} | branch

# Streamlit UI
st.title("Travel Booking Assistance")

# Input from the user
request_text = st.text_input("Enter your travel details:", "I want to book a 7 day trip to Vienna to visit my cousin there.")

# Process the input
if st.button("Check Requirements"):
    # Run the full chain with the input
    response = full_chain.invoke({"request": request_text})
    # Display the response
    st.write("Response:", response)
