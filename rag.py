import google.generativeai as genai
import os
import dotenv
from preprocessing import extract_text_from_pdf
from sentence_transformers import SentenceTransformer
import chromadb
import numpy as np

class RAGService:
    def __init__(self, pdf_path):
        dotenv.load_dotenv()
        GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
        genai.configure(api_key=GOOGLE_API_KEY)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        self.text = extract_text_from_pdf(pdf_path) # Get text directly
        self.chunks = self.chunk_text(self.text)
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.client = chromadb.Client()  # Use in-memory Chroma
        self.collection = self.create_collection()
        self.add_to_collection(self.chunks)

    def chunk_text(self, text, chunk_size=500, overlap=50):
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def create_collection(self):
      collection = self.client.create_collection("my_collection")
      return collection

    def add_to_collection(self, chunks):
        embeddings = self.embedding_model.encode(chunks).tolist()
        ids = [str(i) for i in range(len(chunks))]

        self.collection.add(
            embeddings=embeddings,
            documents=chunks,
            ids=ids
        )

    def retrieve_relevant_chunks(self, query, top_k=3):
        query_embedding = self.embedding_model.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k
        )
        return results['documents'][0]

    def generate_answer(self, query):
        relevant_chunks = self.retrieve_relevant_chunks(query)
        context = "\n".join(relevant_chunks)
        prompt = f"""Answer the following question based on the context provided but don't mention it, keep the tone friendly and warm and answer with confidence:
                    Question: {query}
                    Context:
                    {context}

                    If the answer cannot be found in the context, respond with 'I am sorry, but I don't have enough information to answer that question from the context I was given.'
                    """
        response = self.model.generate_content(prompt)
        return response.text

if __name__ == '__main__':
    pdf_file = "The_Gift_of_the_Magi.pdf"
    rag_service = RAGService(pdf_file)
    user_query = "What did Della sell to buy Jim a gift?"
    answer = rag_service.generate_answer(user_query)
    print(answer)

    user_query = "What is the capital of France?"
    answer = rag_service.generate_answer(user_query)
    print(answer)
