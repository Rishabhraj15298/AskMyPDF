from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 400,
    )
    
    split_docs = []
    for doc in documents : 
        chunks = text_splitter.split_text(doc.page_content)
        for chunk in chunks:
            split_docs.append(Document(
                page_content=chunk, 
                metadata={"page_number": doc.metadata.get("page", None)}
            ))
    return split_docs

        
