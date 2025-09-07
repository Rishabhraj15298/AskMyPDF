from dotenv import load_dotenv
from modules.embeddings import get_embedding_model

from langchain_qdrant import QdrantVectorStore
import os
import google.generativeai as genai
def get_gemini_response(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-pro")
    response = model.generate_content(prompt)
    return response.text if hasattr(response, 'text') else str(response)


load_dotenv()

embedding_model = get_embedding_model()

vector_db = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name = "pdf_collection",
    embedding=embedding_model
)






def generate_answer(query, relevant_chunks):
    context = "\n\n\n".join([
        f"Page {doc.metadata.get('page_number', 'N/A')}: {doc.page_content}"
        for doc in relevant_chunks
    ])
    SYSTEM_PROMPT = (
        "You are a helpful AI Assistant who answers user queries based on the available context "
        "retrieved from a PDF file along with page_contents and page number.\n\n"
        "You should only answer the user based on the following context and navigate the user "
        "to open the right page number to know more.\n\n"
        "Context:\n{context}\n\nQuestion: {query}\nAnswer:"
    )
    prompt = SYSTEM_PROMPT.format(context=context, query=query)
    return get_gemini_response(prompt)






