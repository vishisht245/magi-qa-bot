import google.generativeai as genai
import os
import dotenv
from preprocessing import extract_text_from_pdf
from sentence_transformers import SentenceTransformer
import chromadb

class RAGService:
    def __init__(self, pdf_path):
        """Initializes the RAG service by setting up the generative model, extracting text, and creating a vector database."""
        dotenv.load_dotenv()  # Load environment variables
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))  # Configure Google Gemini API
        self.model = genai.GenerativeModel('gemini-1.5-flash') # Initialize the generative model
        self.text = extract_text_from_pdf(pdf_path)
        self.chunks = self.chunk_text(self.text) # Creating chunks
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2') # Sentence transformer 
        self.client = chromadb.Client() 
        self.collection = self.client.create_collection("my_collection")
        self.add_to_collection(self.chunks)

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Splits text into overlapping chunks for better retrieval."""
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i:i + chunk_size])
            i += chunk_size - overlap  # Move forward with overlap
        return chunks


    def add_to_collection(self, chunks):
        """Adds extracted text chunks along with their embeddings to the ChromaDB collection."""
        embeddings = self.embedding_model.encode(chunks).tolist()
        ids = [str(i) for i in range(len(chunks))]
        self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)

    def retrieve_relevant_chunks(self, query, top_k=3):
        """Retrieves the top K most relevant text chunks from ChromaDB based on the query."""
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
        return results['documents'][0]  # Return the relevant chunks as a list

    def generate_answer(self, query):
        """Generates an answer using the retrieved text chunks and the generative model."""
        relevant_chunks = self.retrieve_relevant_chunks(query)  # Retrieve relevant context
        context = "\n".join(relevant_chunks)  # Combine the retrieved text chunks

        # Create a prompt for the generative model
        prompt = f"""Answer the following question based on the context provided but don't mention it, keep the tone friendly and warm and answer with confidence:
                    Question: {query}
                    Context:
                    {context}

                    If the answer cannot be found in the context, respond with 'I am sorry, but I don't have enough information to answer that question from the context I was given.'
                    """
        
        return self.model.generate_content(prompt).text  # Generate and return the answer

if __name__ == '__main__':
    # Initialize RAGService with the specified PDF file
    rag_service = RAGService("The_Gift_of_the_Magi.pdf")

    # Test the system with sample queries
    print(rag_service.generate_answer("What did Della sell to buy Jim a gift?"))
    print(rag_service.generate_answer("What is the capital of France?"))
