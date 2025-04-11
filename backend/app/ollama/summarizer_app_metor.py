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
import nltk
from nltk.translate.meteor_score import meteor_score

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
                        else:
                            img = page.to_image(resolution=300).original
                            text = pytesseract.image_to_string(img)
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
    def __init__(self, retriever: PrivacyRetriever, model_name='llama3:8b', max_tokens=4096):
        self.retriever = retriever
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        self.system_prompt = """You are a data privacy expert. Using only the provided context from data privacy laws and guidelines, answer the query accurately, concisely, and directly in formal legal language. Do not speculate or include information beyond the context or question."""
    
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
            logger.info(f"Generated response for '{query}': {response['response']}")
            return response['response']
        except Exception as e:
            raise RuntimeError(f"Generation failed: {e}")

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
    
    # RAGAS evaluation
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ollama_llm,
        embeddings=local_embeddings
    )
    
    # METEOR evaluation
    meteor_scores = []
    for answer, ground_truth in zip(answers, ground_truths):
        if answer and ground_truth:
            # Tokenize for METEOR
            answer_tokens = nltk.word_tokenize(answer)
            ground_truth_tokens = nltk.word_tokenize(ground_truth)
            score = meteor_score([ground_truth_tokens], answer_tokens)
            meteor_scores.append(score)
        else:
            meteor_scores.append(0.0)
    
    # Combine results
    combined_result = {
        "ragas": ragas_result,
        "meteor": meteor_scores  # Return list for multiple questions
    }
    return combined_result

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
    
    #For first run only
    # chunks = processor.extract_text_from_pdfs(pdf_paths)
    # retriever = PrivacyRetriever(chunks=chunks, embedder=processor.embedder, persist_directory="./chroma_db")
    
    # For subsequent runs
    retriever = PrivacyRetriever(chunks=[], embedder=processor.embedder, persist_directory="./chroma_db")
    
    rag = PrivacyRAG(retriever)
    
    query = "How to plan an IS Audit as per the RBI guidelines"
    response = rag.generate_response(query)
    logger.info(f"Response: {response}")
    
    test_data = [
        {
            "question": "What are the obligations of Data Fiduciaries under the DPDPA 2023 for processing personal data?",
            "ground_truth": "Under the DPDPA 2023, Data Fiduciaries are obligated to process personal data only in accordance with the provisions of the act and for lawful purposes for which the data principal has given consent, or for certain legitimate uses as referred to in section 7."
        },
        {
            "question": "What defines a 'significant data fiduciary' under DPDPA 2023?",
            "ground_truth": "A 'Significant Data Fiduciary' under DPDPA 2023 is any Data Fiduciary or class of Data Fiduciaries notified by the Central Government under section 10, considering their ability to impact the rights and freedoms of data principals significantly."
        },
        {
            "question": "What are the duties of intermediaries according to the updated IT rules of 2021?",
            "ground_truth": "Intermediaries must publish rules and regulations, privacy policy, and user agreement prominently on their platforms. They should also ensure that users do not host, display, upload, modify, publish, transmit, store, update, or share any information that is harmful, harassing, blasphemous, defamatory, obscene, pornographic, paedophilic, or infringes upon the privacy of others."
        },
        {
            "question": "What is defined as 'online curated content' under the IT rules 2021?",
            "ground_truth": "Online curated content refers to any curated catalog of audio-visual content excluding news and current affairs, owned, licensed, or contracted to be transmitted by a publisher of online curated content and made available on demand via the internet or computer networks."
        },
        {
            "question": "What are the legal consequences for failure to protect data under the IT Act 2000?",
            "ground_truth": "Under section 43A of the IT Act 2000, if a body corporate possessing, dealing, or handling any sensitive personal data fails to implement and maintain reasonable security practices resulting in wrongful loss or wrongful gain to any person, then such body corporate may be liable to pay damages to the affected persons."
        },
        {
            "question": "Define the term 'digital signature' as per the IT Act 2000.",
            "ground_truth": "A digital signature is a type of electronic signature that uses a secure key pair, consisting of a private key for creating the digital signature and a public key to verify it, ensuring the authenticity of electronic records."
        },
        {
            "question": "What are the key components of IT governance according to RBI guidelines?",
            "ground_truth": "Key components of IT governance according to RBI guidelines include strategic alignment of IT with business, value delivery, risk management, resource management, and performance measurement."
        },
        {
            "question": "What does the RBI recommend for the management of IT services outsourcing?",
            "ground_truth": "The RBI recommends that banks should have a robust governance structure for IT services outsourcing, ensuring proper risk management, data confidentiality, service quality, and compliance with regulatory requirements."
        },
        {
            "question": "What does the Right to Information Act 2005 ensure for citizens?",
            "ground_truth": "The Right to Information Act 2005 ensures that all citizens have the right to access information under the control of public authorities to promote transparency and accountability in the working of every public authority."
        },
        {
            "question": "Under the RTI Act 2005, what are the obligations of public authorities regarding the publication of information?",
            "ground_truth": "Public authorities are obliged to maintain records duly cataloged and indexed, publish various particulars of their organization, functions, and duties, and ensure that all records suitable to be computerized are computerized to facilitate the right to information."
        }
    ]
    
    evaluation_results = evaluate_rag(rag, test_data)
    logger.info(f"Evaluation Results - RAGAS: {evaluation_results['ragas']}")
    for i, meteor_score in enumerate(evaluation_results['meteor']):
        logger.info(f"Evaluation Results - METEOR for question {i+1}: {meteor_score}")

if __name__ == "__main__":
    asyncio.run(main())