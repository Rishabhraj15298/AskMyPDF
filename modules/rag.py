from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings , ChatGoogleGenerativeAI

from langchain_qdrant import QdrantVectorStore
import os 


load_dotenv()

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    api_key=os.getenv("GEMINI_API_KEY")
)

vector_db = QdrantVectorStore.from_existing_collection(
    url = "http://localhost:6333",
    collection_name = "pdf_collection",
    embedding=embedding_model
)






def generate_answer  (query , relevant_chunks) :
        
    context = "\n\n\n".join([f"Page {doc.metadata.get('page_number', 'N/A')}: {doc.page_content}" 
                         for doc in relevant_chunks])

    SYSTEM_PROMPT = """
        You are a helpfull AI Assistant who answers user query based on the available context
        retrieved from a PDF file along with page_contents and page number.

        You should only ans the user based on the following context and navigate the user
        to open the right page number to know more.

        Context:
        {context}

        Question:{query}
        Answer:

    """

    # Initialize the chat model
    llm = ChatGoogleGenerativeAI(
        model = "gemini-pro",
        api_key = os.getenv("GEMINI_API_KEY")
    )

    response = llm.invoke(SYSTEM_PROMPT.format(context=context, query=query)   )
    return response






