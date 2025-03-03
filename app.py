import streamlit as st
from rag import RAGService
from summarization import generate_summary
import os  

# Initialize RAG service
@st.cache_resource
def get_rag_service():
    pdf_file = "The_Gift_of_the_Magi.pdf"
    if not os.path.exists(pdf_file):
        st.error(f"Error: PDF file '{pdf_file}' not found.")
        return None
    try:
        return RAGService(pdf_file)
    except Exception as e:
        st.error(f"Error initializing RAG service: {e}")
        return None

rag_service = get_rag_service()

# App UI
st.title("The Gift of the Magi Q&A")
st.write("Ask questions about 'The Gift of the Magi'.")

# Summary section
with st.expander("Show Summary"):
    if rag_service:
        st.write(generate_summary(rag_service.text))
    else:
        st.write("Summary not available.")

# User input and response
user_query = st.text_input("Enter your question:")
if user_query and rag_service:
    with st.spinner("Thinking..."):
        try:
            st.write(rag_service.generate_answer(user_query))
        except Exception as e:
            st.error(f"Error generating answer: {e}")
