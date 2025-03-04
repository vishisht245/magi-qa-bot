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

# Set page configuration
st.set_page_config(
    page_title="The Gift of the Magi Q&A",
    layout="centered"
)

# Initialize RAG service
rag_service = get_rag_service()
if rag_service is None:
    st.error("ERROR: Unable to initialize the RAG service")
    st.stop()
if rag_service.model is None:
    st.error("ERROR: The Google Gemini API key is not set or is invalid")
    st.stop()

# Page header
st.title("The Gift of the Magi Q&A")
st.write("Ask questions about 'The Gift of the Magi'.")

# Summary section
with st.expander("Show Summary"):
    if rag_service and rag_service.text is not None:
        with st.spinner("Generating summary..."):
            summary_text = generate_summary(rag_service.text)
        st.markdown(f"{summary_text}")
    else:
        st.write("Summary not available.")

# Question form
with st.form("query_form"):
    user_query = st.text_input("Enter your question:")
    submitted = st.form_submit_button("Submit")
    
    if submitted:
        user_query = user_query.strip()
        if not user_query:
            st.error("Please enter a question.")
        else:
            with st.spinner("Thinking..."):
                try:
                    answer = rag_service.generate_answer(user_query)
                    st.markdown(f"{answer}")
                except Exception as e:
                    st.error(f"Error generating answer: {e}")


