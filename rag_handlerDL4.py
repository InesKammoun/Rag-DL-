# rag_handlerDL3_fintech.py

import os
import re
import json
import requests
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from pymilvus import connections, MilvusClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
from whoosh.analysis import StemmingAnalyzer
import torch


class RAGHandler:
    """
    Handler RAG pour documents financiers.
    - Docs path par défaut : 'FinTech'
    - LLM : Ollama (local) via API
    - Embedding : SentenceTransformer
    - Reranking : CrossEncoder
    """

    def __init__(self, docs_path="FinTech", milvus_uri=None, index_dir="bm25_index"):
        self.docs_path = docs_path
        self.chunk_collection = "rag_chunks"
        self.hq_collection = "hq_chunks"
        self.index_dir = index_dir

        # Embeddings et reranker
        self.embedding_model = SentenceTransformer("intfloat/e5-large-v2")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Ollama API
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
        self.ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

        # Connexion Milvus
        if milvus_uri is None:
            milvus_uri = os.getenv("MILVUS_URI", "tcp://localhost:19530")
        m = re.match(r"tcp://([^:]+):(\d+)", milvus_uri)
        host, port = (m.group(1), int(m.group(2))) if m else ("localhost", 19530)
        connections.connect(alias="default", host=host, port=port)
        self.milvus_client = MilvusClient()

    # ---------------- Utilities ----------------
    def clean_text(self, text):
        """Nettoie le texte des espaces inutiles."""
        return re.sub(r"\s+", " ", text.strip()) if text else ""

    def emb_text(self, text, is_query=False):
        """Retourne le vecteur normalisé pour un texte."""
        prefix = "query: " if is_query else "passage: "
        emb = self.embedding_model.encode(prefix + text)
        return (emb / np.linalg.norm(emb)).tolist()

    # ---------------- Ollama wrapper ----------------
    def _ollama_generate(self, prompt, max_tokens=256, temperature=0.0):
        """Envoie un prompt à Ollama et retourne la réponse texte."""
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "options": {"max_tokens": max_tokens, "temperature": temperature}
        }

        # Tentative streaming
        try:
            resp = requests.post(self.ollama_url, json=payload, stream=True, timeout=60)
            text = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                try:
                    decoded = json.loads(line.decode("utf-8"))
                    if isinstance(decoded, dict):
                        if "response" in decoded:
                            text += decoded["response"]
                        elif "choices" in decoded:
                            for ch in decoded.get("choices", []):
                                text += ch.get("delta", {}).get("content", "") or ch.get("text", "")
                except Exception:
                    try:
                        text += line.decode("utf-8")
                    except Exception:
                        pass
            if text.strip():
                return text.strip()
        except Exception:
            pass

        # Fallback POST classique
        try:
            resp = requests.post(self.ollama_url, json=payload, timeout=60)
            if resp.ok:
                j = resp.json()
                if "response" in j:
                    return j["response"]
                if "choices" in j:
                    return "".join([c.get("text", "") for c in j["choices"]])
                return str(j)
            else:
                return f"[OLLAMA ERROR] {resp.status_code}: {resp.text}"
        except Exception as e:
            return f"[OLLAMA EXCEPTION] {e}"

    # ---------------- Document loading ----------------
    def _load_all_documents(self):
        """Charge tous les PDF du dossier et les découpe en chunks."""
        splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        chunks = []

        for root, _, files in os.walk(self.docs_path):
            for file in files:
                if file.lower().endswith(".pdf"):
                    path = os.path.join(root, file)
                    try:
                        loader = PyPDFLoader(path)
                        pages = loader.load()
                        split_chunks = splitter.split_documents(pages)
                        for i, chunk in enumerate(split_chunks):
                            chunk.metadata.update({"doc_name": file, "chunk_id": i, "full_path": path})
                            chunks.append(chunk)
                    except Exception as e:
                        print(f"[ERROR] failed to load {path}: {e}")
        return chunks

    # ---------------- Hypothetical questions ----------------
    def _generate_batch_hypothetical_questions(self, chunk_texts):
        """Génère 2 questions hypothétiques par chunk via Ollama."""
        prompt = "Below are document chunks. For each, generate 2 concise hypothetical questions it could answer.\n\n"
        for idx, text in enumerate(chunk_texts):
            prompt += f"Chunk {idx+1}:\n{text}\n\n"
        prompt += "Output format:\nChunk 1:\n- Q1\n- Q2\n..."

        decoded = self._ollama_generate(prompt, max_tokens=256, temperature=0.0)

        # Extraction robuste des questions
        chunk_questions = [[] for _ in chunk_texts]
        blocks = re.findall(r"(Chunk\s+\d+:\s*(?:- .+(?:\n- .+)*)+)", decoded, re.DOTALL | re.IGNORECASE)
        for block in blocks:
            idx_match = re.search(r"Chunk\s+(\d+):", block)
            if idx_match:
                idx = int(idx_match.group(1)) - 1
                qs = re.findall(r"-\s*(.+)", block)
                if 0 <= idx < len(chunk_texts):
                    chunk_questions[idx] = qs[:2]

        # fallback
        for i, text in enumerate(chunk_texts):
            if not chunk_questions[i]:
                chunk_questions[i] = [
                    f"What does this chunk say about {self.clean_text(text)[:60]}?",
                    "What key point can be extracted from this chunk?"
                ]
        return chunk_questions

    # ---------------- Vector store ----------------
    def build_vector_store(self, force_rebuild=False):
        """Construit ou reconstruit les collections Milvus pour les chunks et HQs."""
        if force_rebuild:
            for col in [self.chunk_collection, self.hq_collection]:
                if self.milvus_client.has_collection(col):
                    self.milvus_client.drop_collection(col)

        chunks = self._load_all_documents()
        if not chunks:
            print("Aucun document ou chunk trouvé.")
            return

        dim = len(self.emb_text("test"))

        for col in [self.chunk_collection, self.hq_collection]:
            if not self.milvus_client.has_collection(col):
                self.milvus_client.create_collection(
                    collection_name=col, dimension=dim, metric_type="IP", consistency_level="Strong"
                )

        BATCH_SIZE = 5
        for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Indexing chunks + HQs"):
            batch_chunks = chunks[i:i + BATCH_SIZE]
            chunk_texts = [self.clean_text(c.page_content) for c in batch_chunks]
            chunk_vecs = [self.emb_text(t) for t in chunk_texts]

            # Insert chunks
            chunk_data = [ {
                "id": i + j,
                "vector": vec,
                "text": text,
                "metadata": json.dumps(chunk.metadata)
            } for j, (vec, text, chunk) in enumerate(zip(chunk_vecs, chunk_texts, batch_chunks)) ]
            self.milvus_client.insert(self.chunk_collection, data=chunk_data)

            # Insert HQs
            hq_questions_list = self._generate_batch_hypothetical_questions(chunk_texts)
            hq_data = []
            for j, (questions, chunk, text) in enumerate(zip(hq_questions_list, batch_chunks, chunk_texts)):
                for k, q in enumerate(questions):
                    q_vec = self.emb_text(q, is_query=True)
                    hq_data.append({
                        "id": int(f"{i+j}{k}"),
                        "vector": q_vec,
                        "text": text,
                        "metadata": json.dumps({"question": q, **chunk.metadata})
                    })
            if hq_data:
                self.milvus_client.insert(self.hq_collection, data=hq_data)

    # ---------------- BM25 ----------------
    def build_bm25_index(self):
        if not os.path.exists(self.index_dir):
            os.makedirs(self.index_dir)
        schema = Schema(doc_name=ID(stored=True), content=TEXT(stored=True, analyzer=StemmingAnalyzer()))
        ix = create_in(self.index_dir, schema)
        writer = ix.writer()
        for chunk in self._load_all_documents():
            writer.add_document(doc_name=chunk.metadata["doc_name"], content=self.clean_text(chunk.page_content))
        writer.commit()

    def search_bm25(self, query, top_k=10):
        ix = open_dir(self.index_dir)
        with ix.searcher() as searcher:
            parser = QueryParser("content", ix.schema)
            q = parser.parse(query)
            results = searcher.search(q, limit=top_k)
            return [{"text": r["content"], "score": r.score, "meta": {"doc_name": r["doc_name"]}} for r in results]

    # ---------------- HQ vector search ----------------
    def search_hq_vector(self, query, top_k=10):
        vec = self.emb_text(query, is_query=True)
        results = self.milvus_client.search(
            collection_name=self.hq_collection,
            data=[vec],
            limit=top_k,
            output_fields=["text", "metadata"]
        )[0]
        return [{
            "text": self.clean_text(r["entity"]["text"]),
            "score": r["distance"],
            "meta": json.loads(r["entity"]["metadata"])
        } for r in results]

    # ---------------- Reranking ----------------
    def rerank_results(self, query, passages, top_k=3):
        if not passages:
            return []
        pairs = [[query, p['text']] for p in passages]
        scores = self.reranker.predict(pairs)
        for i, score in enumerate(scores):
            passages[i]['rerank_score'] = float(score)
        return sorted(passages, key=lambda x: -x['rerank_score'])[:top_k]

    # ---------------- Window helpers ----------------
    def sentence_window_retrieval(self, passages, window_size=2, stride=1):
        windows = []
        for i in range(0, len(passages), stride):
            window_texts = passages[i:i + window_size]
            combined = " ".join([p['text'] for p in window_texts])
            if len(combined) <= 1500:
                windows.append({"text": combined})
        return windows

    def adjust_chunks_sorting(self, passages):
        if not passages:
            return passages
        passages_sorted = sorted(passages, key=lambda p: p.get('rerank_score', p.get('score', 0)), reverse=True)
        if len(passages_sorted) < 3:
            return passages_sorted
        best, middle, worst = passages_sorted[0], passages_sorted[1:-1], passages_sorted[-1]
        middle_sorted = sorted(middle, key=lambda p: p.get('rerank_score', p.get('score', 0)))
        return [best] + middle_sorted + [worst]

    # ---------------- Complexity / Subqueries ----------------
    def is_complex_question(self, query, length_threshold=15):
        if len(query.split()) > length_threshold:
            return True
        keywords = [" and ", " or ", "difference", "compare", ",", "steps", "workflow", "vs", "versus"]
        return any(k in query.lower() for k in keywords)

    def generate_subqueries(self, query, max_subq=3):
        prompt = f"Break this complex question into simpler sub-questions:\n\n{query}\n\nSub-questions:"
        decoded = self._ollama_generate(prompt, max_tokens=128, temperature=0.0)
        subq = re.findall(r'[-\d\*]\s*(.+)', decoded)
        return subq[:max_subq] if subq else [query]

    # ---------------- Hybrid pipeline ----------------
    def hybrid_pipeline(self, query, top_k=10, final_k=3, use_windows=True, window_size=2, stride=1):
        queries = [query]
        if self.is_complex_question(query):
            print(f"[⚠️] Complex query detected. Generating sub-questions.")
            queries = self.generate_subqueries(query)

        all_passages = []
        for subquery in queries:
            with ThreadPoolExecutor(max_workers=2) as executor:
                f_bm25 = executor.submit(self.search_bm25, subquery, top_k)
                f_hqvec = executor.submit(self.search_hq_vector, subquery, top_k)
                bm25_results, vector_results = f_bm25.result(), f_hqvec.result()

            combined = {r['text']: r for r in vector_results}
            for r in bm25_results:
                combined.setdefault(r['text'], r)

            reranked = self.rerank_results(subquery, list(combined.values()), top_k=final_k)
            all_passages.extend(reranked)

        all_passages = sorted(all_passages, key=lambda x: -x.get('rerank_score', 0))

        if use_windows:
            windows = self.sentence_window_retrieval(all_passages, window_size, stride)
            adjusted = self.adjust_chunks_sorting(windows)
            return {"query": query, "results": adjusted, "raw_reranked": all_passages}
        else:
            return {"query": query, "results": all_passages[:final_k], "raw_reranked": all_passages}

    # ---------------- Generate answer ----------------
    def generate_answer(self, query, top_k=10, final_k=3, use_windows=True, window_size=2, stride=1):
        """
        Génère une réponse claire et concise à partir du pipeline RAG.
        - Les phrases sont séparées par des points, pas de *, / ou \n.
        - Maximum 3 phrases.
        """
        output = self.hybrid_pipeline(query, top_k, final_k, use_windows, window_size, stride)
        top_context = output["raw_reranked"][0]["text"] if output["raw_reranked"] else ""

        prompt = f"""
Answer the question below in plain English in 3 short sentences.
- Do NOT use bullets (*, -, etc.), slashes (/), or newline characters.
- Do NOT copy text verbatim from the context.
- Keep sentences concrete and concise.

Question:
{query}

Context:
{top_context}

Answer:
"""

        decoded = self._ollama_generate(prompt, max_tokens=200, temperature=0.0)

        # Nettoyage : suppression de *, /, \n, espaces superflus
        text = decoded.replace("\n", " ")
        text = re.sub(r"[\*\-\•]", "", text)
        text = re.sub(r"\s+", " ", text).strip()

        # Limiter à 3 phrases (approx.)
        sentences = re.split(r'(?<=[.!?]) +', text)
        answer = " ".join(sentences[:3])

        return {
            "query": query,
            "answer": answer,
            "context": top_context
        }
