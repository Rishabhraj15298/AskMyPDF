import streamlit as st
from modules import embeddings, ingestion, rag, vector_db

st.set_page_config(page_title="AskMyPDF", page_icon="📄")

st.title("📄 AskMyPDF")
st.write("Upload your PDF and ask questions about its content!")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file is not None:
    st.success("File uploaded successfully!")

    with st.spinner("Processing the PDF..."):
        # Step 1: Ingest the PDF and split into chunks
        text_chunks = ingestion.process_pdf(uploaded_file)

        # Step 2: Get the embedding model
        embedding_model = embeddings.get_embedding_model()

        # Step 3: Create/load vector DB and insert chunks
        qdrant = vector_db.store_in_qdrant(text_chunks, embedding_model)

        # Step 4: Ask questions
        query = st.text_input("Ask a question about the PDF:")
        if query:
            with st.spinner("Generating answer..."):
                # Retrieve relevant chunks
                relevant_chunks = qdrant.similarity_search(query, k=3)

                # Generate the answer using RAG
                answer = rag.generate_answer(query, relevant_chunks)

                st.write("**Answer:**")
                st.write(answer)
else:
    st.info("Please upload a PDF file to get started.")
