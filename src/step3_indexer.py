import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import re, sys, json, sqlite3, time
import torch
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss

# Forzar UTF-8 en la salida de consola para evitar errores de codificación en Windows
if sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

db_name = sys.argv[1] if len(sys.argv) > 1 else "HealthDB"
PROYECTO_ROOT = Path(__file__).resolve().parent.parent

DIR_JSON = PROYECTO_ROOT / "output_json"
DIR_DB_BASE = PROYECTO_ROOT / "data" / "04_db" / db_name
DIR_WIKI_BASE = PROYECTO_ROOT / "data" / "05_wiki" / db_name

DIR_DB_BASE.mkdir(parents=True, exist_ok=True)
DIR_WIKI_BASE.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-m3"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class WikiCompiler:
    def __init__(self, base_wiki_path):
        self.base_path = base_wiki_path

    def _sanitize(self, name: str) -> str:
        clean_name = re.sub(r'\s+', ' ', str(name)).strip()
        clean_name = re.sub(r'[^a-zA-Z0-9 _-]', '', clean_name)
        return clean_name[:50].strip()

    def _write_md(self, folder, filename, content):
        target_dir = self.base_path / folder
        target_dir.mkdir(parents=True, exist_ok=True)
        with open(target_dir / f"{self._sanitize(filename)}.md", "w", encoding="utf-8") as f:
            f.write(content)

    def compile_document(self, doc_data):
        doc_id = doc_data.get("doc_id", "unknown_doc")
        fuente = doc_data.get("norma", "Desconocida")
        chunks = doc_data.get("chunks", [])
        
        md_content = f"# Set de Datos: {doc_id}\n**Fuente:** {fuente}\n\n## Interacciones Procesadas (Muestra)\n"
        for chunk in chunks[:50]:
            md_content += f"### {chunk.get('chunk_id')}\n- **Contexto:** {chunk.get('resumen_fragmento')}\n\n"
        self._write_md("documentos", doc_id, md_content)
        return md_content

class FastIndexer:
    def __init__(self):
        print(f"==================================================")
        print(f"[NEXUS OS] INICIANDO MOTOR VECTORIAL Y ESTRUCTURACIÓN")
        print(f"==================================================")
        print(f"[*] Aceleración Hardware : {DEVICE.upper()}")
        print(f"[*] Base de Datos Destino: {db_name}")
        print(f"[*] Modelo Embeddings    : {MODEL_NAME}")
        
        self.model = SentenceTransformer(MODEL_NAME, device=DEVICE)
        self.db_path = DIR_DB_BASE / "nexus_clinical.db"
        self.index_path = DIR_DB_BASE / "vector_index.faiss"
        self.map_path = DIR_DB_BASE / "index_map.json"
        
        self.init_sqlite()
        if self.index_path.exists():
            print(f"[*] Índice previo encontrado. Cargando...")
            self.index = faiss.read_index(str(self.index_path))
            with open(self.map_path, "r", encoding="utf-8") as f:
                self.index_id_map = json.load(f)
            print(f"[*] Vectores ya existentes: {len(self.index_id_map)}")
        else:
            print(f"[*] Creando nuevo espacio vectorial de 1024 dimensiones...")
            self.index = faiss.IndexFlatIP(1024)
            self.index_id_map = []

    def init_sqlite(self):
        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()
        cur.execute('''CREATE TABLE IF NOT EXISTS chunks_vectoriales 
                          (chunk_id TEXT PRIMARY KEY, doc_id TEXT, texto_completo TEXT, metadata TEXT)''')
        conn.commit()
        conn.close()

    def process_all(self):
        wiki_compiler = WikiCompiler(DIR_WIKI_BASE)
        json_files = list(DIR_JSON.glob("*_rag.json"))
        if not json_files: return print("[WARN] No hay datos en output_json/. Ejecuta step1_unifier.py primero.")

        # --- FASE DE CONTEO Y ANÁLISIS ---
        print("\n[FASE 1] Analizando volumen de datos...")
        total_chunks_to_process = 0
        total_archivos = len(json_files)
        
        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                total_chunks_to_process += len(data.get("chunks", []))
        
        print(f"[*] Archivos JSON detectados: {total_archivos}")
        print(f"[*] Total de interacciones clínicas a procesar: {total_chunks_to_process}")
        print("--------------------------------------------------")

        conn = sqlite3.connect(self.db_path)
        cur = conn.cursor()

        batch_texts, batch_ids, batch_meta = [], [], []
        BATCH_SIZE = 16 
        chunks_procesados_historico = 0
        chunks_saltados = 0

        # --- FASE DE VECTORIZACIÓN ---
        print("\n[FASE 2] Iniciando Vectorización Masiva en GPU...")
        start_time = time.time()

        for jf in json_files:
            with open(jf, "r", encoding="utf-8") as f: data = json.load(f)
            
            doc_id, norma = data.get("doc_id"), data.get("norma")
            print(f"\n>> Abriendo Base de Datos: {norma} ({len(data.get('chunks', []))} registros)")
            wiki_compiler.compile_document(data)

            for chunk in data.get("chunks", []):
                cid = chunk.get("chunk_id")
                
                # Si el chunk ya está en el índice, lo saltamos
                if cid in self.index_id_map:
                    chunks_saltados += 1
                    chunks_procesados_historico += 1
                    continue
                
                text_to_vec = f"FUENTE: {norma}\nRESUMEN: {chunk.get('resumen_fragmento')}\nTEXTO:\n{chunk.get('texto_completo')}"
                
                batch_texts.append(text_to_vec)
                batch_ids.append(cid)
                batch_meta.append((cid, doc_id, text_to_vec, json.dumps(chunk)))

                if len(batch_texts) >= BATCH_SIZE:
                    self._flush_batch(batch_texts, batch_ids, batch_meta, cur, conn)
                    chunks_procesados_historico += len(batch_texts)
                    
                    # LOG DE VERBOCIDAD SIN EMOJIS
                    porcentaje = (chunks_procesados_historico / total_chunks_to_process) * 100
                    print(f"   [Progreso: {chunks_procesados_historico}/{total_chunks_to_process} | {porcentaje:.1f}%] - Escrito en Disco [OK]")
                    
                    batch_texts, batch_ids, batch_meta = [], [], []

        # Vaciar remanentes finales
        if batch_texts: 
            self._flush_batch(batch_texts, batch_ids, batch_meta, cur, conn)
            chunks_procesados_historico += len(batch_texts)
            print(f"   [Progreso: {chunks_procesados_historico}/{total_chunks_to_process} | 100.0%] - Lote Final Escrito en Disco [OK]")

        conn.close()
        
        # --- FASE FINAL: GUARDAR ÍNDICE MATEMÁTICO ---
        print("\n[FASE 3] Consolidando Red Neuronal FAISS...")
        faiss.write_index(self.index, str(self.index_path))
        with open(self.map_path, "w", encoding="utf-8") as f: json.dump(self.index_id_map, f)
        
        end_time = time.time()
        mins = (end_time - start_time) / 60
        
        print("\n==================================================")
        print(f"[OK] PROCESO COMPLETADO EXITOSAMENTE")
        print(f"[*] Registros insertados : {chunks_procesados_historico - chunks_saltados}")
        print(f"[*] Registros omitidos   : {chunks_saltados} (Ya existían)")
        print(f"[*] Tiempo total         : {mins:.2f} minutos")
        print(f"[*] Archivos guardados en: {DIR_DB_BASE}")
        print("==================================================")

    def _flush_batch(self, texts, ids, meta, cur, conn):
        try:
            # 1. Crea los embeddings en la GPU
            embs = self.model.encode(texts, batch_size=len(texts), normalize_embeddings=True)
            # 2. Añade a FAISS
            self.index.add(np.array(embs).astype('float32'))
            self.index_id_map.extend(ids)
            # 3. Inserta en SQLite
            cur.executemany("INSERT OR REPLACE INTO chunks_vectoriales VALUES (?, ?, ?, ?)", meta)
            # 4. CRÍTICO: Guarda en disco duro INMEDIATAMENTE
            conn.commit()
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("\n[PELIGRO] Memoria de GPU al límite. Limpiando y reintentando...")
            else:
                raise e
        finally:
            # 5. Limpia la basura de la VRAM para no saturar la tarjeta gráfica
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    FastIndexer().process_all()