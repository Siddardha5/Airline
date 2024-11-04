import os
import streamlit as st
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OpenAIkey"]

# Create the LLM API object
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

#Source: https://www.perplexity.ai/search/import-os-import-streamlit-as-.d.JoOBWRA66L32dUbg9.w?utm_source=backtoschool


# Define the positive experience chain
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


branch = RunnableBranch(
    (lambda x: "negative" in x["feedback_type"].lower() and "airline fault" in x["airline_fault"].lower(), negative_airline_fault_chain),
    (lambda x: "negative" in x["feedback_type"].lower() and "not airline fault" in x["airline_fault"].lower(), negative_not_airline_fault_chain),
    (lambda x: "positive" in x["feedback_type"].lower(), positive_chain),
    general_chain
)

#Source: https://api.python.langchain.com/en/latest/_modules/langchain_core/runnables/branch.html
#Source: https://api.python.langchain.com/en/latest/runnables/langchain_core.runnables.branch.RunnableBranch.html
#Source: https://python.langchain.com/v0.1/docs/expression_language/primitives/functions/


# Streamlit app setup
st.title("Airline Experience Feedback")

# Get user input for feedback
feedback = st.text_area("Share your experience of the latest trip with us.")

if st.button("Submit"):
    feedback_type = "positive" if "good" in feedback.lower() or "great" in feedback.lower() else "negative"
    airline_fault = "airline fault" if "lost luggage" in feedback.lower() or "delay by airline" in feedback.lower() else "not airline fault"

    # Get the appropriate response using branching logic
    response = branch.invoke({"feedback_type": feedback_type, "airline_fault": airline_fault, "feedback": feedback})

    # Display the response
    st.write(response)

