import os
import streamlit as st
from langchain_core.runnables import RunnableBranch
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.output_parsers import StrOutputParser

# OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

# Create the LLM API object
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the sentiment analysis template for determining if feedback is positive or negative
sentiment_template = """You are a professional customer service representative skilled in handling customer issues.
From the following text, determine whether the feedback is positive or negative.

Respond with only one word: "positive" or "negative".

Text:
{feedback}
"""

# Create the sentiment decision-making chain
sentiment_chain = (
    PromptTemplate.from_template(sentiment_template)
    | llm
    | StrOutputParser()
)

# Define the positive experience response chain
positive_chain = PromptTemplate.from_template(
    """You are a professional customer service representative.
    The customer has shared a positive experience with the airline. Respond professionally, thanking them for their feedback and for choosing to fly with the airline.

    Your response should follow these guidelines:
    1. Address the customer directly and to the point. 
    2. Appreciate for their positive feedback.
    3. Do not respond with any reasoning. Just respond professionally as a professional customer service representative.
    4. Keep the response encouraging and professional, encouraging them to choose the airline again in the future.

Text:
{feedback}
"""
) | llm

# Define the negative experience chain for issues caused by the airline
negative_airline_fault_chain = PromptTemplate.from_template(
    """You are a professional customer service representative skilled in handling customer issues.
    The customer had a negative experience due to an issue caused by the airline (e.g., lost luggage). Offer your sympathies, inform the customer that customer service will reach out soon to resolve the issue or provide compensation.

    Your response should follow these guidelines:
    1. Address the customer directly, to the point and express sincere apologies for the inconvenience.
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
    1. Address the customer directly, to the point and apologize for the inconvenience they experienced.
    2. Politely explain that the situation was beyond the airline's control, and express appreciation for their understanding.
    3. Keep the tone empathetic and professional.

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
    # Use the sentiment_chain to determine if feedback is positive or negative
    feedback_type = sentiment_chain.invoke({"feedback": feedback}).strip().lower()
    
    # Use the airline fault template only if feedback is negative
    airline_fault = ""
    if feedback_type == "negative":
        airline_fault_template = """You are a professional customer service representative skilled in handling customer issues.
        From the following text, determine if the negative experience was caused by the airline (e.g., lost luggage) or by an external factor (e.g., weather-related delay).

        Respond with only one word: "airline fault" or "not airline fault".

        Text:
        {feedback}
        """
        airline_fault_chain = (
            PromptTemplate.from_template(airline_fault_template)
            | llm
            | StrOutputParser()
        )
        airline_fault = airline_fault_chain.invoke({"feedback": feedback}).strip().lower()

    # Get the appropriate response using branching logic
    response = branch.invoke({"feedback_type": feedback_type, "airline_fault": airline_fault, "feedback": feedback})

    # Display the response
    st.write(response)
