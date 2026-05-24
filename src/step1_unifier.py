import json
import hashlib
from pathlib import Path

def crear_chunk(fuente, doc_id, num_art, resumen, texto):
    # PREVENCIÓN DE ERROR: Convertir explícitamente a string por si el JSON trae 'null'
    res_str = str(resumen) if resumen is not None else "Sin resumen disponible"
    txt_str = str(texto) if texto is not None else "Sin contenido"
    
    chunk_id = hashlib.md5(txt_str.encode('utf-8')).hexdigest()[:12]
    return {
        "chunk_id": f"{fuente[:3]}_{chunk_id}",
        "numero_articulo": str(num_art),
        "resumen_fragmento": res_str[:150],
        "texto_completo": txt_str,
        "temas": [fuente],
        "articulos_citados": []
    }

def unificar_datos():
    PROYECTO_ROOT = Path(__file__).resolve().parent.parent
    OUT_DIR = PROYECTO_ROOT / "output_json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. HealthQA
    med_path = PROYECTO_ROOT / "medical.json"
    med_chunks = []
    if med_path.exists():
        with open(med_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for i, item in enumerate(data):
                txt = f"PACIENTE: {item.get('input_text','')}\nCONTEXTO: {item.get('context','')}\nDOC: {item.get('output_text','')}"
                med_chunks.append(crear_chunk("HealthQA", "MedQA", f"QA-{i}", item.get("context",""), txt))
        with open(OUT_DIR / "health_rag.json", "w", encoding="utf-8") as f:
            json.dump({"doc_id": "HealthQA", "norma": "HealthQA", "nivel": 1, "chunks": med_chunks}, f)

    # 2. Diseases
    dis_path = PROYECTO_ROOT / "DiseasesOutput.json"
    dis_chunks = []
    if dis_path.exists():
        with open(dis_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                txt = f"ENFERMEDAD: {item.get('text','')} ({item.get('name','')})\nDESC: {item.get('laytext','')}\nTIPO: {item.get('category','')}\nICD10: {item.get('ICD10','')}"
                dis_chunks.append(crear_chunk("Diseases", "Enfermedades", item.get('name',''), item.get('text',''), txt))
        with open(OUT_DIR / "diseases_rag.json", "w", encoding="utf-8") as f:
            json.dump({"doc_id": "Diseases", "norma": "Diseases", "nivel": 2, "chunks": dis_chunks}, f)

    # 3. Symptoms
    sym_path = PROYECTO_ROOT / "SymptomsOutput.json"
    sym_chunks = []
    if sym_path.exists():
        with open(sym_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for item in data:
                txt = f"SÍNTOMA/SIGNO: {item.get('text','')}\nLAYTEXT: {item.get('laytext','')}\nCATEGORÍA: {item.get('category','')}\nTIPO: {item.get('type','')}"
                sym_chunks.append(crear_chunk("Symptoms", "Sintomas", item.get('name',''), item.get('laytext',''), txt))
        with open(OUT_DIR / "symptoms_rag.json", "w", encoding="utf-8") as f:
            json.dump({"doc_id": "Symptoms", "norma": "Symptoms", "nivel": 3, "chunks": sym_chunks}, f)

    # 4. Dictionary
    dic_path = PROYECTO_ROOT / "medical_dictionary.json"
    dic_chunks = []
    if dic_path.exists():
        with open(dic_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            for i, item in enumerate(data.get("definitions", [])):
                txt = f"TÉRMINO: {item.get('term','')}\nDEFINICIÓN: {item.get('definition','')}"
                dic_chunks.append(crear_chunk("Dictionary", "Diccionario", f"DEF-{i}", item.get('term',''), txt))
        with open(OUT_DIR / "dictionary_rag.json", "w", encoding="utf-8") as f:
            json.dump({"doc_id": "Dictionary", "norma": "Dictionary", "nivel": 4, "chunks": dic_chunks}, f)

    print("[OK] Las 4 bases de datos fueron unificadas exitosamente en output_json/")

if __name__ == "__main__":
    unificar_datos()