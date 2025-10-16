import pandas as pd
import numpy as np
from datasets import Dataset
import google.generativeai as genai
import requests
from typing import List, Dict, Tuple

# RAGas imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy, 
        context_precision,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    )
    from langchain_google_genai import ChatGoogleGenerativeAI
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

class RAGEvaluator:
    """
    Classe pour évaluer un système RAG avec RAGas
    """
    
    def __init__(self, google_api_key: str, api_base_url: str = "http://localhost:8000"):
        self.google_api_key = google_api_key
        self.api_base_url = api_base_url
        self.available = RAGAS_AVAILABLE
        
        # Configure Google AI
        genai.configure(api_key=self.google_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        if self.available:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                google_api_key=self.google_api_key,
                temperature=0
            )
    
    def check_api_availability(self) -> bool:
        """Vérifie si l'API RAG est disponible"""
        try:
            response = requests.get(f"{self.api_base_url}/ping", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def fetch_documents(self, query: str, num_docs: int) -> List[Dict]:
        """Récupère des documents de la base de données"""
        try:
            response = requests.post(
                f"{self.api_base_url}/search",
                json={"query": query, "top_k": num_docs * 2, "final_k": num_docs},
                timeout=60
            )
            
            if response.status_code == 200:
                return response.json().get("results", [])
            else:
                raise Exception(f"API Error: {response.status_code}")
        except Exception as e:
            raise Exception(f"Failed to fetch documents: {e}")
    
    def generate_ground_truth_qa(self, documents: List[Dict], num_questions: int = 10) -> List[Dict]:
        """Génère des paires Q&A de vérité terrain à partir des documents"""
        qa_pairs = []
        
        for i, doc in enumerate(documents[:num_questions]):
            prompt = f"""Based on this FinTech document, generate 1 high-quality question and its accurate answer:

Document: {doc['text'][:1000]}

Generate:
1. A specific question about the content
2. A detailed, accurate answer based only on the document

Format:
Question: [your question]
Answer: [your answer]"""

            try:
                response = self.model.generate_content(prompt)
                result = response.text.strip()
                
                lines = result.split('\n')
                question = ""
                answer = ""
                
                for line in lines:
                    if line.startswith("Question:"):
                        question = line.replace("Question:", "").strip()
                    elif line.startswith("Answer:"):
                        answer = line.replace("Answer:", "").strip()
                
                if question and answer:
                    qa_pairs.append({
                        "question": question,
                        "ground_truth": answer,
                        "contexts": [doc['text']]
                    })
            except Exception as e:
                print(f"Error generating Q&A for document {i}: {e}")
        
        return qa_pairs
    
    def get_rag_answers(self, questions: List[str], contexts: List[List[str]], 
                       top_k: int = 5, final_k: int = 3, window_size: int = 1) -> Tuple[List[str], List[List[str]]]:
        """Obtient les réponses du système RAG"""
        answers = []
        retrieved_contexts = []
        
        for question, context in zip(questions, contexts):
            try:
                # Get answer
                response = requests.post(
                    f"{self.api_base_url}/answer",
                    json={
                        "query": question,
                        "top_k": top_k,
                        "final_k": final_k,
                        "window_size": window_size
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    if "application/json" in response.headers.get("content-type", ""):
                        result = response.json()
                        answer = result.get("answer", "No answer")
                    else:
                        answer = response.text.strip()
                else:
                    answer = "Error in RAG system"
                
                # Get contexts
                search_response = requests.post(
                    f"{self.api_base_url}/search",
                    json={"query": question, "top_k": final_k, "final_k": final_k},
                    timeout=30
                )
                
                if search_response.status_code == 200:
                    search_results = search_response.json().get("results", [])
                    contexts_list = [r.get("text", "") for r in search_results]
                else:
                    contexts_list = context
                    
            except Exception as e:
                answer = f"Error: {e}"
                contexts_list = context
            
            answers.append(answer)
            retrieved_contexts.append(contexts_list)
        
        return answers, retrieved_contexts
    
    def evaluate_system(self, qa_pairs: List[Dict], top_k: int = 5, 
                       final_k: int = 3, window_size: int = 1) -> Tuple[Dict, Dict]:
        """Évalue le système RAG avec les métriques RAGas"""
        
        if not self.available:
            raise Exception("RAGas not available. Install: pip install ragas datasets langchain-google-genai")
        
        questions = [qa["question"] for qa in qa_pairs]
        ground_truths = [qa["ground_truth"] for qa in qa_pairs]
        original_contexts = [qa["contexts"] for qa in qa_pairs]
        
        # Get RAG answers
        answers, retrieved_contexts = self.get_rag_answers(
            questions, original_contexts, top_k, final_k, window_size
        )
        
        # Create dataset
        data = {
            "question": questions,
            "answer": answers,
            "contexts": retrieved_contexts,
            "ground_truth": ground_truths
        }
        
        dataset = Dataset.from_dict(data)
        
        # Evaluate
        result = evaluate(
            dataset=dataset,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                context_relevancy,
                answer_correctness,
                answer_similarity
            ],
            llm=self.llm,
            embeddings=None  # Will use default
        )
        
        return result, data
    
    def run_full_evaluation(self, test_query: str = "fintech blockchain cryptocurrency banking", 
                           num_questions: int = 10, top_k: int = 5, 
                           final_k: int = 3, window_size: int = 1) -> Dict:
        """Lance une évaluation complète"""
        
        # Check API
        if not self.check_api_availability():
            raise Exception("RAG API not available")
        
        # Fetch documents
        documents = self.fetch_documents(test_query, num_questions)
        
        # Generate Q&A
        qa_pairs = self.generate_ground_truth_qa(documents, num_questions)
        
        # Evaluate
        result, data = self.evaluate_system(qa_pairs, top_k, final_k, window_size)
        
        return {
            "evaluation_result": result,
            "evaluation_data": data,
            "qa_pairs": qa_pairs,
            "num_documents": len(documents),
            "num_qa_pairs": len(qa_pairs)
        }
    
    def format_results_for_display(self, evaluation_result) -> List[Dict]:
        """Formate les résultats pour l'affichage"""
        metrics_data = []
        for metric, score in evaluation_result.items():
            if isinstance(score, (int, float)) and not np.isnan(score):
                metrics_data.append({"Metric": metric, "Score": f"{score:.4f}"})
        return metrics_data
    
    def get_score_interpretation(self, score: float) -> str:
        """Interprète un score"""
        if score >= 0.8:
            return "🟢 Excellent"
        elif score >= 0.6:
            return "🟡 Good"
        elif score >= 0.4:
            return "🟠 Needs Improvement"
        else:
            return "🔴 Poor"