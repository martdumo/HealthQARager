import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
import sys, json, time, sqlite3
from pathlib import Path
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

def main():
    if len(sys.argv) < 3:
        print("Uso: python step5_feedback.py <db_name> <json_path>")
        return

    db_name = sys.argv[1]
    json_path = sys.argv[2]

    if not os.path.exists(json_path):
        print(f"[ERROR] No se encuentra el archivo temporal: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    query = data.get("query", "")
    report = data.get("report", "")

    PROYECTO_ROOT = Path(__file__).resolve().parent.parent
    db_dir = PROYECTO_ROOT / "data" / "04_db" / db_name
    index_path = db_dir / "vector_index.faiss"
    map_path = db_dir / "index_map.json"
    sqlite_path = db_dir / "jurisprudencia.db"

    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model = SentenceTransformer("BAAI/bge-m3", device=device)

    index = faiss.read_index(str(index_path))
    with open(map_path, "r", encoding="utf-8") as f:
        index_map = json.load(f)
    
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()

    text_to_vec = f"CONSULTA CLÍNICA PREVIA: {query}\nSÍNTESIS MÉDICA IA: {report}"
    embedding = model.encode([text_to_vec], normalize_embeddings=True)
    vector = np.array(embedding).astype('float32')

    chunk_id = f"CLINICAL_SYNTHESIS_{int(time.time())}"

    index.add(vector)
    index_map.append(chunk_id)

    metadata = {"norma": "Síntesis Clínica RAG", "numero_articulo": "Registro Interno"}
    
    cursor.execute("""
        INSERT INTO chunks_vectoriales (chunk_id, doc_id, numero_articulo, texto_completo, metadata, is_wiki)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (chunk_id, "SINTESIS_MEDICA", "REGISTRO", report, json.dumps(metadata), 0))

    faiss.write_index(index, str(index_path))
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(index_map, f)
    
    conn.commit()
    conn.close()

    if os.path.exists(json_path):
        os.remove(json_path)
    
    print("[OK] Conocimiento clínico asimilado exitosamente en la DB Vectorial")

if __name__ == "__main__":
    main()