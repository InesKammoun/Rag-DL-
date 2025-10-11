# main.py
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel
from rag_handlerDL4  import RAGHandler  

# === App Initialization ===
app = FastAPI(title="🔍 HQ-based RAG Pipeline API")
router = APIRouter()

# === Init RAG ===
DOCS_PATH = "./FinTech"
print("🚀 Initializing RAGHandler...")
rag = RAGHandler(docs_path=DOCS_PATH, milvus_uri="tcp://localhost:19530")
print("✅ RAGHandler initialized.")

# === Request Model ===
class QueryModel(BaseModel):
    query: str
    top_k: int = 5
    final_k: int = 3
    window_size: int = 1  # taille de la fenêtre, 0 ou 1 = désactivé

# === Routes ===
@router.get("/")
async def root():
    return {"message": "🚀 Welcome to the HQ-based RAG API. Try /ping, /rebuild, /search, /answer"}

@router.get("/ping")
async def ping():
    return {"status": "✅ API is running"}

@router.post("/rebuild")
async def rebuild_indexes():
    try:
        print("🔄 Rebuilding vector stores (chunks + HQs) and BM25 index...")
        rag.build_vector_store(force_rebuild=True)
        rag.build_bm25_index()
        print("✅ All indexes rebuilt successfully.")
        return {"status": "success", "message": "Indexes rebuilt successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/search")
async def search(query: QueryModel):
    try:
        print(f"[🔎] Query received: {query.query}")
        output = rag.hybrid_pipeline(
            query=query.query,
            top_k=query.top_k,
            final_k=query.final_k,
            window_size=query.window_size
        )
        return {
            "query": query.query,
            "top_k": query.top_k,
            "results": [
                {
                    "text": r["text"],
                    "score": r.get("score"),
                    "rerank_score": r.get("rerank_score")
                }
                for r in output["raw_reranked"]
            ]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@router.post("/answer")
async def answer(query: QueryModel):
    try:
        print(f"[🧠] Generating answer for query: {query.query}")
        answer_text = rag.generate_answer(
            query=query.query,
            top_k=query.top_k,
            final_k=query.final_k,
            window_size=query.window_size
        )
        return {
            "query": query.query,
            "answer": answer_text
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# === Mount router ===
app.include_router(router)
