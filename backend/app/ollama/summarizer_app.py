import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import chromadb
import pdfplumber
import pytesseract
from PIL import Image
from transformers import LlamaTokenizer
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Tuple
from chromadb.api.types import EmbeddingFunction
from datasets import Dataset
import logging
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom Embedding Function for Chroma
class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.embedder.encode(input, show_progress_bar=False)
        return embeddings.tolist()

# 1. PDF Processing and Chunking with OCR
class DataPrivacyProcessor:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L12-v2')  # Consider upgrading to 'all-MiniLM-L12-v2'
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        
    def extract_text_from_pdfs(self, pdf_paths: List[str]) -> List[str]:
        all_chunks = []
        for pdf_path in pdf_paths:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text and len(text.strip()) > 10:
                            chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk) > 50]
                            all_chunks.extend(chunks)
                        else:
                            img = page.to_image(resolution=300).original
                            text = pytesseract.image_to_string(img)
                            chunks = [chunk.strip() for chunk in text.split('\n\n') if len(chunk) > 50]
                            all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
        return all_chunks

# 2. Retrieval System with Chroma and Reranking
class PrivacyRetriever:
    def __init__(self, chunks: List[str], embedder: SentenceTransformer, persist_directory: str = "./chroma_db"):
        self.chunks = chunks
        self.embedder = embedder
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')  # For reranking
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = SentenceTransformerEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name="privacy_chunks",
            embedding_function=self.embedding_function
        )
        
        if self.collection.count() == 0 and chunks:
            embeddings = self.embedder.encode(chunks, show_progress_bar=True)
            self.collection.add(
                ids=[f"chunk_{i}" for i in range(len(chunks))],
                embeddings=embeddings.tolist(),
                metadatas=[{"text": chunk} for chunk in chunks]
            )
        
    def retrieve(self, query: str, top_k: int = 10, min_similarity: float = 0.5) -> List[Tuple[str, float]]:
        try:
            query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2  # Get more candidates for reranking
            )
            
            # Initial retrieval with similarity filter
            candidates = []
            for text, distance in zip(results["metadatas"][0], results["distances"][0]):
                similarity = 1 - distance
                if similarity >= min_similarity:
                    candidates.append((text["text"], similarity))
            
            if not candidates:
                logger.warning(f"No chunks above min_similarity {min_similarity} for query: {query}")
                return []
            
            # Rerank using cross-encoder
            chunk_texts = [chunk for chunk, _ in candidates]
            rerank_scores = self.reranker.predict([(query, chunk) for chunk in chunk_texts])
            reranked = sorted(zip(chunk_texts, rerank_scores), key=lambda x: x[1], reverse=True)[:top_k]
            
            retrieved = [(chunk, float(score)) for chunk, score in reranked]
            logger.info(f"Retrieved {len(retrieved)} chunks for query '{query}': {[f'{chunk[:50]}... ({score:.3f})' for chunk, score in retrieved]}")
            return retrieved
            
        except ValueError as ve:
            raise ValueError(f"Query encoding failed: {ve}")
        except Exception as e:
            raise RuntimeError(f"Retrieval failed unexpectedly: {e}")

# 3. RAG with LLaMA3:8B
class PrivacyRAG:
    def __init__(self, retriever: PrivacyRetriever, model_name='llama3:8b', max_tokens=4096):
        self.retriever = retriever
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        self.system_prompt = """You are a data privacy expert. Using the provided context from data privacy laws and guidelines, answer the query accurately and concisely in formal legal language."""
    
    def _truncate_context(self, context: str, query: str) -> str:
        prompt_tokens = len(self.tokenizer.encode(self.system_prompt))
        query_tokens = len(self.tokenizer.encode(query))
        available_tokens = self.max_tokens - prompt_tokens - query_tokens - 100
        context_tokens = self.tokenizer.encode(context)
        if len(context_tokens) > available_tokens:
            context_tokens = context_tokens[:available_tokens]
            return self.tokenizer.decode(context_tokens)
        return context
    
    def generate_response(self, query: str) -> str:
        try:
            context_chunks_with_scores = self.retriever.retrieve(query)
            context_chunks = [chunk for chunk, _ in context_chunks_with_scores]
            context = "\n\n".join(context_chunks)
            context = self._truncate_context(context, query)
            full_prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuery:\n{query}"
            response = ollama.generate(model=self.model_name, prompt=full_prompt)
            return response['response']
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

# 4. RAGAS Evaluation with Ollama
def evaluate_rag(rag: PrivacyRAG, test_data: List[dict]) -> dict:
    ollama_llm = OllamaLLM(model="llama3:8b")
    local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    questions = [item['question'] for item in test_data]
    ground_truths = [item['ground_truth'] for item in test_data]
    contexts = []
    answers = []
    
    for q in questions:
        context_chunks_with_scores = rag.retriever.retrieve(q)
        context_chunks = [chunk for chunk, _ in context_chunks_with_scores]
        if not context_chunks:
            logger.warning(f"No relevant chunks retrieved for question: {q}")
        contexts.append(context_chunks)
        answer = rag.generate_response(q)
        answers.append(answer if answer else "No response generated")
    
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })
    
    result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ollama_llm,
        embeddings=local_embeddings
    )
    return result

# Async main function
async def main():
    processor = DataPrivacyProcessor()
    pdf_paths = [
        "./data-privacy-pdf/RBI-Guidelines.pdf", 
        "./data-privacy-pdf/4.-CBPR-Policies-Rules-and-Guidelines-Revised-For-Posting-3-16-updated-1709-2019.pdf", 
        "./data-privacy-pdf/16MC9102DB7D5FE742CCB5D0715A77F6666E.pdf", 
        "./data-privacy-pdf/2024-0118-Policy-SEBI_Circular_on_Cybersecurity_and_Cyber_Resilience_Framework_(CSCRF)_for_SEBI_Regulated.pdf", 
        "./data-privacy-pdf/20240905112741.pdf", 
        "./data-privacy-pdf/Aadhaar_Act_2016_as_amended.pdf", 
        "./data-privacy-pdf/book_indiacybersecurity.pdf", 
        "./data-privacy-pdf/DPDPA - 2023.pdf", 
        "./data-privacy-pdf/EPRS_ATA(2020)659275_EN.pdf", 
        "./data-privacy-pdf/GBS300411F.pdf", 
        "./data-privacy-pdf/health_management_policy_bac9429a79.pdf", 
        "./data-privacy-pdf/in098en.pdf", 
        "./data-privacy-pdf/Information-Technology-Intermediary-Guidelines-and-Digital-Media-Ethics-Code-Rules-2021-updated-06.04.202.pdf", 
        "./data-privacy-pdf/it_act_2000_updated.pdf", 
        "./data-privacy-pdf/Legal Framework for Data Protection and Security and Privacy norms.pdf", 
        "./data-privacy-pdf/Personal Data Protection Bill, 2019.pdf", 
        "./data-privacy-pdf/Privacy and Data Protection.pdf", 
        "./data-privacy-pdf/rti-act.pdf", 
        "./data-privacy-pdf/Takshashila_07_11_2017.pdf"
    ]
    # chunks = processor.extract_text_from_pdfs(pdf_paths)
    # retriever = PrivacyRetriever(chunks=chunks, embedder=processor.embedder, persist_directory="./chroma_db")
    
    # For subsequent runs
    retriever = PrivacyRetriever(chunks=[], embedder=processor.embedder, persist_directory="./chroma_db")
    rag = PrivacyRAG(retriever)
    
    query = "How to plan an IS Audit as per the RBI guidelines"
    response = rag.generate_response(query)
    logger.info(f"Response: {response}")
    
    test_data = [
    #{
    #    "question": "What are the penalties for data privacy violations under DPDPA 2023?",
    #    "ground_truth": "Under DPDPA 2023, penalties can include fines up to INR 250 crore per instance for significant breaches."
    #},
    #{
    #    "question": "Does DPDPA 2023 allow for imprisonment in cases of non-compliance?",
    #    "ground_truth": "DPDPA 2023 does not specify imprisonment as a penalty for non-compliance."
    #},
    #{
    #    "question": "Under DPDPA 2023, are companies liable for data breaches caused by third-party service providers?",
    #    "ground_truth": "Yes, under DPDPA 2023, companies can be held liable for data breaches caused by third-party service providers, depending on the circumstances and contractual obligations."
    #},

    ]
    evaluation_results = evaluate_rag(rag, test_data)
    logger.info(f"RAGAS Results: {evaluation_results}")

if __name__ == "__main__":
    asyncio.run(main())