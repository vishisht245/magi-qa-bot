import google.generativeai as genai
import os
import dotenv
from preprocessing import extract_text_from_pdf
from sentence_transformers import SentenceTransformer
import chromadb

class RAGService:
    def __init__(self, pdf_path):
        """Initializes the RAG service."""
        dotenv.load_dotenv()
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.text = extract_text_from_pdf(pdf_path)
        self.chunks = self.chunk_text(self.text)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # PersistentClient for database persistence
        self.client = chromadb.PersistentClient("./chroma_db")
        
        # Check if the collection exists; create it if it doesn't
        try:
            # This is the updated part for ChromaDB v0.6.0
            collection_names = self.client.list_collections()
            if "my_collection" not in collection_names:
                self.collection = self.client.create_collection("my_collection")
                self.add_to_collection(self.chunks)
            else:
                self.collection = self.client.get_collection("my_collection")
        except Exception as e:
            print(f"Error initializing ChromaDB: {e}")
            raise

    def chunk_text(self, text, chunk_size=500, overlap=50):
        """Splits text into overlapping chunks."""
        chunks, i = [], 0
        while i < len(text):
            chunks.append(text[i:i + chunk_size])
            i += chunk_size - overlap  # Move forward with overlap
        return chunks

    def add_to_collection(self, chunks):
        """Adds text chunks and embeddings to ChromaDB."""
        try:
            embeddings = self.embedding_model.encode(chunks).tolist()
            ids = [str(i) for i in range(len(chunks))]
            self.collection.add(embeddings=embeddings, documents=chunks, ids=ids)
        except Exception as e:
            print(f"Error adding to collection: {e}")
            raise

    def retrieve_relevant_chunks(self, query, top_k=3):
        """Retrieves relevant chunks from ChromaDB."""
        try:
            query_embedding = self.embedding_model.encode([query]).tolist()
            results = self.collection.query(query_embeddings=query_embedding, n_results=top_k)
            return results['documents'][0]
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return ["There was an error retrieving information from the database."]

    def generate_answer(self, query):
        """Generates an answer using retrieved chunks and Gemini."""
        try:
            relevant_chunks = self.retrieve_relevant_chunks(query)
            context = "\n".join(relevant_chunks)
            prompt = f"""Answer the following question strictly based on the context provided **only** but don't mention it, keep the tone natural and warm and answer with confidence:
             If the answer cannot be found in the context, respond with 'I am sorry, but I don't have enough information to answer that question from the context I was given.'
             
             Question: {query}
             Context:
            {context}
             """
            return self.model.generate_content(prompt).text
        except Exception as e:
            print(f"Error generating answer: {e}")
            return f"I encountered an error while trying to answer your question. Error details: {str(e)}"

if __name__ == '__main__':
    rag_service = RAGService("The_Gift_of_the_Magi.pdf")
    print(rag_service.generate_answer("What did Della sell to buy Jim a gift?"))
    print(rag_service.generate_answer("What is the capital of India?"))
