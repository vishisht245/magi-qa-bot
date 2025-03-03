import streamlit as st
from rag import RAGService
from summarization import generate_summary
# No need to import preprocessing directly here

# Initialize RAG service (cache it!)
pdf_file = "The_Gift_of_the_Magi.pdf"

@st.cache_resource
def get_rag_service():
    return RAGService(pdf_file)

rag_service = get_rag_service()

# App UI
st.title("The Gift of the Magi Q&A")
st.write("Ask questions about 'The Gift of the Magi'.")

# Summary section
with st.expander("Show Summary"):
    st.write(generate_summary(rag_service.text)) # Directly use rag_service.text

# User input
user_query = st.text_input("Enter your question:")

# Generate and display answer
if user_query:
    with st.spinner("Thinking..."):
        answer = rag_service.generate_answer(user_query)
    st.write(answer)