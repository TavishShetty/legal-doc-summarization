import streamlit as st
from anonymize import anonymize_pdf
from summarize import DataPrivacyProcessor, PrivacyRetriever, PrivacyRAG
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app
st.title("Data Privacy & Anonymization Demo")

# Tabbed interface
tab1, tab2 = st.tabs(["PDF Anonymization", "Privacy Q&A"])

with tab1:
    st.header("Upload a PDF to Anonymize")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_file:
        # Save uploaded file temporarily
        temp_pdf = "temp.pdf"
        with open(temp_pdf, "wb") as f:
            f.write(uploaded_file.getbuffer())
        if st.button("Anonymize"):
            output_path = "anonymized_output.txt"
            try:
                anonymize_pdf(temp_pdf, output_path)
                with open(output_path, "r", encoding="utf-8") as f:
                    anonymized_text = f.read()
                st.text_area("Anonymized Text", anonymized_text, height=300)
                st.download_button(
                    "Download Anonymized Text",
                    anonymized_text,
                    file_name="anonymized_output.txt",
                    mime="text/plain"
                )
            except Exception as e:
                st.error(f"Error during anonymization: {e}")
            finally:
                # Clean up temporary files
                if os.path.exists(temp_pdf):
                    os.remove(temp_pdf)
                if os.path.exists(output_path):
                    os.remove(output_path)

with tab2:
    st.header("Ask a Data Privacy Question")
    query = st.text_input(
        "Enter your question",
        placeholder="e.g., What are the obligations of Data Fiduciaries under DPDPA 2023?"
    )
    if st.button("Get Answer"):
        try:
            # Initialize RAG with precomputed ChromaDB
            processor = DataPrivacyProcessor()
            retriever = PrivacyRetriever(chunks=[], embedder=processor.embedder, persist_directory="./chroma_db")
            rag = PrivacyRAG(retriever)
            with st.spinner("Generating response..."):
                response = rag.generate_response(query)
            st.write("**Answer:**")
            st.write(response)
        except Exception as e:
            st.error(f"Error generating response: {e}")