import os
import streamlit as st
from langchain_core.runnables import RunnableBranch
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

# Create the LLM API object
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the sentiment analysis template for determining if feedback is positive or negative
sentiment_template = PromptTemplate.from_template(
    """You are a professional customer service representative skilled in handling customer issues.
    From the following text, determine whether the feedback is positive or negative.

    Respond with only one word: "positive" or "negative".

Text:
{feedback}
"""
) | llm

# Define the airline fault detection template for determining if the negative feedback was due to the airline or external factors
airline_fault_template = PromptTemplate.from_template(
    """You are a professional customer service representative skilled in handling customer issues.
    From the following text, determine if the negative experience was caused by the airline (e.g., lost luggage) or by an external factor (e.g., weather-related delay).

    Respond with only one word: "airline fault" or "not airline fault".

Text:
{feedback}
"""
) | llm

# Define the positive experience response chain
positive_chain = PromptTemplate.from_template(
    """Thank you for choosing our airline. We're thrilled to hear about your positive experience!
    We appreciate your feedback and look forward to serving you again.

Text:
{feedback}
"""
) | llm

# Define the negative experience response chain for issues caused by the airline
negative_airline_fault_chain = PromptTemplate.from_template(
    """We’re very sorry for the inconvenience caused by the airline. Our customer service team will contact you soon to resolve the issue or provide compensation.

Text:
{feedback}
"""
) | llm

# Define the negative experience response chain for non-airline issues
negative_not_airline_fault_chain = PromptTemplate.from_template(
    """We’re sorry to hear about the inconvenience. Unfortunately, this issue was beyond the airline's control. Thank you for your understanding.

Text:
{feedback}
"""
) | llm

# Define a fallback response chain for cases where none of the conditions match
general_chain = PromptTemplate.from_template(
    """Thank you for sharing your experience. We value your feedback.

Text:
{feedback}
"""
) | llm

# Define branching logic based on sentiment and fault detection
branch = RunnableBranch(
    (lambda x: x["feedback_type"] == "negative" and x["airline_fault"] == "airline fault", negative_airline_fault_chain),
    (lambda x: x["feedback_type"] == "negative" and x["airline_fault"] == "not airline fault", negative_not_airline_fault_chain),
    (lambda x: x["feedback_type"] == "positive", positive_chain),
    general_chain
)

# Streamlit app setup
st.title("Airline Experience Feedback")

# Get user input for feedback
feedback = st.text_area("Share your experience of the latest trip with us.")

if st.button("Submit"):
    # Use the sentiment template to determine if feedback is positive or negative
    feedback_type = sentiment_template.invoke({"feedback": feedback}).strip().lower()
    
    # Use the airline fault template only if feedback is negative
    airline_fault = ""
    if feedback_type == "negative":
        airline_fault = airline_fault_template.invoke({"feedback": feedback}).strip().lower()

    # Get the appropriate response using branching logic
    response = branch.invoke({"feedback_type": feedback_type, "airline_fault": airline_fault, "feedback": feedback})

    # Display the response
    st.write(response)
