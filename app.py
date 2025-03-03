#app.py
import streamlit as st
from rag import RAGService
from summarization import generate_summary
import os

@st.cache_resource
def get_rag_service():
    """Initializes and caches the RAGService."""
    pdf_file = "The_Gift_of_the_Magi.pdf"
    if not os.path.exists(pdf_file):
        st.error(f"Error: PDF file '{pdf_file}' not found.")
        return None
    try:
        return RAGService(pdf_file)
    except Exception as e:
        st.error(f"Error initializing RAGService: {e}")
        return None

rag_service = get_rag_service()

# --- Check for API Key Error BEFORE UI ---
if rag_service is None or rag_service.model is None:  # Check for model as well
    st.error("ERROR: The Google Gemini API key is not set or is invalid")
    st.stop()

# App UI
st.title("The Gift of the Magi Q&A")
st.write("Ask questions about 'The Gift of the Magi'.")

# Summary section
with st.expander("Show Summary"):
    if rag_service and rag_service.text is not None:
        st.write(generate_summary(rag_service.text))
    else:
        st.write("Summary not available.")


# User input and response
user_query = st.text_input("Enter your question:")

if user_query: # Check if user enter the query
    with st.spinner("Thinking..."):
        # Removed the try...except block
        answer = rag_service.generate_answer(user_query)  # Call directly
        st.write(answer) # Display the result
