import ollama
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
import pdfplumber
import pytesseract
from PIL import Image
from transformers import LlamaTokenizer
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
    def __init__(self, model_name: str = "all-MiniLM-L12-v2"):
        self.embedder = SentenceTransformer(model_name)
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.embedder.encode(input, show_progress_bar=False)
        return embeddings.tolist()

class DataPrivacyProcessor:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L12-v2')
        self.tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b")
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)  # Increased size
        
    def extract_text_from_pdfs(self, pdf_paths: List[str]) -> List[str]:
        all_chunks = []
        for pdf_path in pdf_paths:
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    full_text = ""
                    for page in pdf.pages:
                        text = page.extract_text()
                        if text and len(text.strip()) > 10:
                            full_text += text + "\n"
                        else:
                            img = page.to_image(resolution=300).original
                            text = pytesseract.image_to_string(img)
                            full_text += text + "\n"
                    chunks = self.splitter.split_text(full_text)
                    all_chunks.extend([chunk.strip() for chunk in chunks if len(chunk) > 50])
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {e}")
        return all_chunks

class PrivacyRetriever:
    def __init__(self, chunks: List[str], embedder: SentenceTransformer, persist_directory: str = "./chroma_db"):
        self.chunks = chunks
        self.embedder = embedder
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.client = chromadb.PersistentClient(path=persist_directory)
        self.embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L12-v2")
        self.collection = self.client.get_or_create_collection(
            name="privacy_chunks",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"}
        )
        
        if self.collection.count() == 0 and chunks:
            embeddings = self.embedder.encode(chunks, show_progress_bar=True)
            self.collection.add(
                ids=[f"chunk_{i}" for i in range(len(chunks))],
                embeddings=embeddings.tolist(),
                metadatas=[{"text": chunk} for chunk in chunks]
            )
            logger.info(f"Stored {self.collection.count()} chunks in ChromaDB")
            sample_chunks = self.collection.peek(5)
            logger.info(f"Sample chunks: {sample_chunks['documents']}")
        
    def retrieve(self, query: str, top_k: int = 10, min_similarity: float = 0.3) -> List[Tuple[str, float]]:
        try:
            query_embedding = self.embedder.encode([query], show_progress_bar=False).tolist()[0]
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2
            )
            
            candidates = []
            distances = results["distances"][0]
            similarities = [1 - d for d in distances]
            logger.debug(f"Top similarities for query '{query}': {sorted(similarities, reverse=True)[:5]}")
            for text, distance in zip(results["metadatas"][0], distances):
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
    local_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
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
    
    ragas_result = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=ollama_llm,
        embeddings=local_embeddings
    )
    
    meteor_scores = []
    for answer, ground_truth in zip(answers, ground_truths):
        if answer and ground_truth:
            answer_tokens = nltk.word_tokenize(answer)
            ground_truth_tokens = nltk.word_tokenize(ground_truth)
            score = meteor_score([ground_truth_tokens], answer_tokens)
            meteor_scores.append(score)
        else:
            meteor_scores.append(0.0)
    
    return {"ragas": ragas_result, "meteor": meteor_scores}

async def main():
    processor = DataPrivacyProcessor()
    pdf_paths = [
        "./data-privacy-pdf/RBI-Guidelines.pdf",
        "./data-privacy-pdf/2024-0118-Policy-SEBI_Circular_on_Cybersecurity_and_Cyber_Resilience_Framework_(CSCRF)_for_SEBI_Regulated.pdf",
        "./data-privacy-pdf/Aadhaar_Act_2016_as_amended.pdf", 
        "./data-privacy-pdf/DPDPA - 2023.pdf", 
        "./data-privacy-pdf/it_act_2000_updated.pdf", 
        "./data-privacy-pdf/Personal Data Protection Bill, 2019.pdf",
        "./data-privacy-pdf/rti-act.pdf"
    ]
    
    chunks = processor.extract_text_from_pdfs(pdf_paths)
    retriever = PrivacyRetriever(chunks=chunks, embedder=processor.embedder, persist_directory="./chroma_db")
    rag = PrivacyRAG(retriever)
    
    query = "How to plan an IS Audit as per the RBI guidelines"
    response = rag.generate_response(query)
    logger.info(f"Response: {response}")
    
    test_data = [
        {
            "question": "What are Data Fiduciary duties under the DPDPA 2023?",  # Broadened
            "ground_truth": "Under the DPDPA 2023, a Data Fiduciary must process personal data only for lawful purposes with the data principal’s consent or for legitimate uses as per section 7, ensure data accuracy, implement security safeguards, and notify data breaches."
        },
        {
            "question": "What security practices does Section 43A of the IT Act 2000 require?",  # Slightly rephrased
            "ground_truth": "Section 43A of the IT Act 2000 mandates that body corporates handling sensitive personal data implement reasonable security practices and procedures to prevent wrongful loss or gain, with liability for compensation if they fail."
        },
        {
            "question": "How does the RBI guideline recommend planning an IS audit?",  # Simplified
            "ground_truth": "RBI guidelines require banks to plan IS audits to evaluate IT risks, security controls, compliance with regulations, and system integrity, involving risk assessment, scope definition, and periodic reviews."
        },
        {
            "question": "What disclosure obligations do public authorities have under the RTI Act 2005?",
            "ground_truth": "The RTI Act 2005 mandates public authorities to proactively disclose organizational details, functions, and duties, maintain indexed records, and computerize suitable records to facilitate public access."
        },
        {
            "question": "How does the Aadhaar Act 2016 protect biometric data?",
            "ground_truth": "The Aadhaar Act 2016 restricts biometric data use to authentication by authorized entities, prohibits unauthorized sharing, and requires robust security measures to protect Aadhaar data."
        },
        {
            "question": "What cybersecurity measures does SEBI’s CSCRF framework require?",
            "ground_truth": "SEBI’s CSCRF framework mandates regulated entities to conduct risk assessments, implement incident response plans, provide security training, and ensure continuous monitoring to enhance cybersecurity."
        },
        {
            "question": "What are the data localization requirements in the Personal Data Protection Bill, 2019?",
            "ground_truth": "The Personal Data Protection Bill, 2019, requires sensitive personal data to be stored in India, with cross-border transfers allowed only with explicit consent and adequate protection in the recipient country."
        },
        {
            "question": "What penalties does the DPDPA 2023 impose for data protection non-compliance?",
            "ground_truth": "The DPDPA 2023 imposes penalties on Data Fiduciaries for non-compliance, including fines for failing to protect personal data or breaching processing obligations, as specified in the Act."
        },
        {
            "question": "What is the IT Act 2000’s stance on intermediary liability for third-party content?",
            "ground_truth": "Under the IT Act 2000, intermediaries are exempt from liability for third-party content if they act as conduits and comply with due diligence requirements, but lose exemption if they initiate or modify the content."
        },
        {
            "question": "What governance does RBI recommend for IT outsourcing in banks?",
            "ground_truth": "RBI recommends banks establish robust governance for IT outsourcing, including risk management, data confidentiality, service-level agreements, and compliance with regulatory standards."
        }
    ]
    
    evaluation_results = evaluate_rag(rag, test_data)
    logger.info(f"Evaluation Results - RAGAS: {evaluation_results['ragas']}")
    for i, meteor_score in enumerate(evaluation_results['meteor']):
        logger.info(f"Evaluation Results - METEOR for question {i+1}: {meteor_score}")

if __name__ == "__main__":
    asyncio.run(main())