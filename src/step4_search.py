import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sqlite3
import json
from pathlib import Path
import sys

def buscar_rag(db_name, query_text, k=200):
    PROYECTO_ROOT = Path(__file__).resolve().parent.parent
    db_dir = PROYECTO_ROOT / "data" / "04_db" / db_name
    sqlite_path = db_dir / "nexus_clinical.db"
    faiss_path = db_dir / "vector_index.faiss"
    map_path = db_dir / "index_map.json"

    if not faiss_path.exists(): return []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer('BAAI/bge-m3', device=device)
    query_vector = model.encode([query_text], normalize_embeddings=True)
    
    index = faiss.read_index(str(faiss_path))
    distances, indices = index.search(np.array(query_vector).astype('float32'), k)
    
    with open(map_path, 'r', encoding='utf-8') as f:
        index_map = json.load(f)
        
    conn = sqlite3.connect(str(sqlite_path))
    cursor = conn.cursor()
    
    resultados = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1: continue
        try: chunk_id = index_map[int(idx)]
        except IndexError: continue
        
        cursor.execute("SELECT texto_completo, metadata FROM chunks_vectoriales WHERE chunk_id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            texto, meta_raw = row
            meta = json.loads(meta_raw) if meta_raw else {}
            resultados.append({
                "score": float(dist),
                "chunk_id": chunk_id,
                "texto": texto,
                "fuente": meta.get("temas", ["Desconocida"])[0] # Tema = Fuente
            })
            
    conn.close()
    return resultados

if __name__ == "__main__":
    db_arg = sys.argv[1]
    query = sys.argv[2]
    k_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    hits = buscar_rag(db_arg, query, k_arg)
    print(json.dumps(hits))