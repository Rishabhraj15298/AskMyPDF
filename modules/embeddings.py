from langchain_google_genai import GoogleGenAIEmbeddings

def get_embedding_model():
    return GoogleGenAIEmbeddings(
        model="models/gemini-embedding-001"
    )
