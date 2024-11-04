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

st.title("Airline Customer Engagement App")
feedback = st.text_area("Share with us your experience of the latest trip.")

# Define the template to detect weather its positive or negative

feedback_type_template = """You are a sentiment analysis expert. 
From the following text, determine if the experience described is positive or negative.

Do not respond with more than one word.

Text:
{feedback}
"""
airline_fault_template = """You are an expert in airline customer service. 
From the following text, determine if the cause of dissatisfaction is the airline's fault (e.g., lost luggage, flight delay due to staffing issues) or beyond its control (e.g., weather-related delay).

Respond with "airline fault" if the issue is the airline's fault, and "not airline fault" if it is beyond their control.

Text:
{feedback}
"""

feedback_type_chain = (
    PromptTemplate.from_template(feedback_type_template)
    | llm
    | StrOutputParser()
)

airline_fault_chain = (
    PromptTemplate.from_template(airline_fault_template)
    | llm
    | StrOutputParser()
)


positive_chain = PromptTemplate.from_template(
    """You are a professional customer service representative.
    The customer has shared a positive experience with the airline. Respond professionally, thanking them for their feedback and for choosing to fly with the airline.

    Your response should follow these guidelines:
    1. Address the customer directly and express appreciation for their positive feedback.
    2. Keep the response warm and professional, encouraging them to choose the airline again in the future.

Text:
{feedback}
"""
) | llm

# Define the negative experience chain for issues caused by the airline
negative_airline_fault_chain = PromptTemplate.from_template(
    """You are a customer service representative skilled in handling customer grievances.
    The customer had a negative experience due to an issue caused by the airline (e.g., lost luggage). Offer your sympathies, inform the customer that customer service will reach out soon to resolve the issue or provide compensation.

    Your response should follow these guidelines:
    1. Address the customer directly and express sincere apologies for the inconvenience.
    2. Reassure the customer that the airline's customer service team will contact them to resolve the issue or provide compensation.
    3. Keep the tone empathetic and professional.

Text:
{feedback}
"""
) | llm

# Define the negative experience chain for issues beyond the airline's control
negative_not_airline_fault_chain = PromptTemplate.from_template(
    """You are a professional customer service representative.
    The customer had a negative experience due to an issue beyond the airline's control (e.g., weather-related delays). Offer your sympathies, and explain that the airline is not liable in such situations, but appreciate their understanding.

    Your response should follow these guidelines:
    1. Address the customer directly and apologize for the inconvenience they experienced.
    2. Politely explain that the situation was beyond the airline's control, and express appreciation for their understanding.
    3. Keep the tone empathetic and professional.

Text:
{feedback}
"""
) | llm

# Routing/Branching chain
branch = RunnableBranch(
    (lambda x: "negative" in x["feedback_type"].lower() and "airline fault" in x["airline_fault"].lower(),
        lambda _: negative_airline_fault_response,
    ),
    (lambda x: "negative" in x["feedback_type"].lower() and "not airline fault" in x["airline_fault"].lower(),
        lambda _: negative_not_airline_fault_response,
    ),
    (lambda x: "positive" in x["feedback_type"].lower(),
        lambda _: positive_response,
    ),
)

# Combine chains into the final flow
if st.button("Submit"):
    # Pass the feedback through the chains
    feedback_type = feedback_type_chain.invoke({"feedback": feedback})
    if "negative" in feedback_type.lower():
        airline_fault = airline_fault_chain.invoke({"feedback": feedback})
    else:
        airline_fault = "none"  # No need to check fault for positive feedback
    
    # Branch logic based on feedback_type and airline_fault
    response = branch.invoke({"feedback_type": feedback_type, "airline_fault": airline_fault})
    
    # Display the response
    st.write(response)
