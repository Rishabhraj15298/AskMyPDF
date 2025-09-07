from langchain_qdrant import Qdrant
from qdrant_client import QdrantClient

def store_in_qdrant(chunks , embedding_model ):
    client = QdrantClient(host="localhost", port=6333)

    qdrant = Qdrant.from_documents(
        documents = chunks ,
        embedding=embedding_model,
        collection_name="pdf_collection",
        client = client
    )
    return qdrant
