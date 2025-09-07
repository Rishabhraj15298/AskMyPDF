
# Simple Gemini embedding function using google-generativeai
import os
import google.generativeai as genai

def get_embedding_model():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment.")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("models/gemini-embedding-001")
    def embed(text):
        response = model.embed_content([text])
        return response['embedding'] if 'embedding' in response else response
    return embed
