import requests
import numpy as np
import chromadb
import pdfplumber
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import LlamaTokenizer
from typing import List, Tuple
import logging
import nltk
from nltk.translate.meteor_score import meteor_score
from chromadb.api.types import EmbeddingFunction
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from datasets import Dataset
import os
import time

# Download required NLTK data
nltk.download('wordnet')
nltk.download('punkt_tab')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingFunction(EmbeddingFunction):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.embedder.encode(input, show_progress_bar=False)
        return embeddings.tolist()

class DataPrivacyProcessor:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L12-v2')
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
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
        return all_chunks

class PrivacyRetriever:
    def __init__(self, chunks: List[str], embedder: SentenceTransformer, persist_directory: str = "./chroma_db"):
        self.chunks = chunks
        self.embedder = embedder
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
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
        
    def retrieve(self, query: str, top_k: int = 10, min_similarity: float = 0.6) -> List[Tuple[str, float]]:
        try:
            query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2
            )
            
            candidates = []
            for text, distance in zip(results["metadatas"][0], results["distances"][0]):
                similarity = 1 - distance
                if similarity >= min_similarity:
                    candidates.append((text["text"], similarity))
            
            if not candidates:
                logger.warning(f"No chunks above min_similarity {min_similarity} for query: {query}")
                return []
            
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

class PrivacyRAG:
    def __init__(self, retriever, max_tokens=4096):
        self.retriever = retriever
        self.max_tokens = max_tokens
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        self.system_prompt = """You are a data privacy expert. Using only the provided context from data privacy laws and guidelines, answer the query accurately, concisely, and directly in formal legal language. Do not speculate or include information beyond the context or question."""
        self.hf_api_token = os.getenv("HUGGINGFACE_API_TOKEN", "REMOVED")
        self.hf_api_url = "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
        self.headers = {"Authorization": f"Bearer {self.hf_api_token}"}

    def _truncate_context(self, context: str, query: str) -> str:
        prompt_tokens = len(self.tokenizer.encode(self.system_prompt))
        query_tokens = len(self.tokenizer.encode(query))
        available_tokens = self.max_tokens - prompt_tokens - query_tokens - 100
        context_tokens = self.tokenizer.encode(context)
        if len(context_tokens) > available_tokens:
            context_tokens = context_tokens[:available_tokens]
            return self.tokenizer.decode(context_tokens)
        return context

    def query_hf_api(self, payload, max_retries=3, delay=2):
        for attempt in range(max_retries):
            try:
                response = requests.post(self.hf_api_url, headers=self.headers, json=payload)
                response.raise_for_status()
                return response.json()
            except requests.exceptions.HTTPError as e:
                if response.status_code == 503:
                    print(f"API unavailable (503), retrying {attempt + 1}/{max_retries}...")
                    time.sleep(delay)
                else:
                    print(f"API error: {e}")
                    return None
            except Exception as e:
                print(f"Unexpected error: {e}")
                return None
        print("Max retries reached. Falling back to default behavior.")
        return None

    def generate_response(self, query: str) -> str:
        try:
            context_chunks_with_scores = self.retriever.retrieve(query)
            context_chunks = [chunk for chunk, _ in context_chunks_with_scores]
            context = "\n\n".join(context_chunks)
            context = self._truncate_context(context, query)
            full_prompt = f"{self.system_prompt}\n\nContext:\n{context}\n\nQuery:\n{query}"
            payload = {
                "inputs": full_prompt,
                "parameters": {
                    "max_length": 200,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            response = self.query_hf_api(payload)
            if response and isinstance(response, list) and "generated_text" in response[0]:
                answer = response[0]["generated_text"].replace(full_prompt, "").strip()
                logger.info(f"Generated response for '{query}': {answer}")
                return answer
            else:
                logger.error("Invalid response from Hugging Face API")
                return "Error: Unable to generate response."
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"