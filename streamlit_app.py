from langchain_core.runnables import RunnableBranch
from langchain_core.prompts import PromptTemplate
from langchain.llms import OpenAI
import os
import streamlit as st

# Assuming you have the OpenAI key set in your Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

# Define the LLM API object
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define the positive experience chain
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

# Define a fallback response chain for cases where none of the conditions match
fallback_chain = PromptTemplate.from_template(
    """Thank you for sharing your experience. We value your feedback.

Text:
{feedback}
"""
) | llm

# Define the RunnableBranch without a default argument
branch = RunnableBranch(
    (lambda x: "negative" in x["feedback_type"].lower() and "airline fault" in x["airline_fault"].lower(), negative_airline_fault_chain),
    (lambda x: "negative" in x["feedback_type"].lower() and "not airline fault" in x["airline_fault"].lower(), negative_not_airline_fault_chain),
    (lambda x: "positive" in x["feedback_type"].lower(), positive_chain),
    (lambda x: True, fallback_chain)  # Catch-all branch as a fallback
)

# Streamlit app setup
st.title("Airline Experience Feedback")

# Get user input for feedback
feedback = st.text_area("Share with us your experience of the latest trip.")

if st.button("Submit"):
    # Simulate feedback type and fault detection for demonstration
    # In a real scenario, these would come from a model that detects sentiment and fault type
    feedback_type = "positive" if "good" in feedback.lower() or "great" in feedback.lower() else "negative"
    airline_fault = "airline fault" if "lost luggage" in feedback.lower() or "delay by airline" in feedback.lower() else "not airline fault"

    # Get the appropriate response using branching logic
    response = branch.invoke({"feedback_type": feedback_type, "airline_fault": airline_fault, "feedback": feedback})

    # Display the response
    st.write(response)
