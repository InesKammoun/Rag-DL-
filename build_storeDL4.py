# build_store.py
from rag_handlerDL4 import RAGHandler

if __name__ == "__main__":
    print("🚀 Lancement de la construction de l’index Milvus + BM25...")

    # Initialisation du RAGHandler
    rag = RAGHandler(docs_path="FinTech", milvus_uri="tcp://localhost:19530")

    # Construction de l'index vectoriel Milvus (chunks + HQs)
    rag.build_vector_store(force_rebuild=True)
    print("✅ Index vectoriel Milvus (chunks + HQs) construit avec succès.")

    # Construction de l'index BM25
    rag.build_bm25_index()
    print("✅ Index BM25 construit avec succès.")

    # Optionnel : vérifier le nombre de vecteurs
    try:
        chunk_count = rag.milvus_client.count_entities(rag.chunk_collection)
        print(f"[info] Nombre de vecteurs chunks : {chunk_count}")
        # Si HQ collection existe
        if rag.milvus_client.has_collection(rag.hq_collection):
            hq_count = rag.milvus_client.count_entities(rag.hq_collection)
            print(f"[info] Nombre de vecteurs HQ : {hq_count}")
    except Exception as e:
        print(f"[warn] Impossible de compter les vecteurs : {e}")

    print("🎉 Indexation terminée !")
