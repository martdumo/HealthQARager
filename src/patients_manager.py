import json
from pathlib import Path

PROYECTO_ROOT = Path(__file__).resolve().parent.parent
PATIENTS_DIR = PROYECTO_ROOT / "data" / "patients"
PATIENTS_DIR.mkdir(parents=True, exist_ok=True)

def guardar_paciente(datos):
    dni = datos.get("DNI", "00000000")
    nombre = datos.get("Nombre", "Desconocido").replace(" ", "_")
    filename = f"{dni}_{nombre}.json"
    with open(PATIENTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(datos, f, ensure_ascii=False, indent=4)
    return filename

def cargar_pacientes():
    pacientes = []
    for filepath in PATIENTS_DIR.glob("*.json"):
        with open(filepath, "r", encoding="utf-8") as f:
            pacientes.append(json.load(f))
    return pacientes

def paciente_a_texto(datos):
    if not datos: return ""
    txt = "=== HISTORIA CLÍNICA DEL PACIENTE ===\n"
    for categoria, campos in datos.items():
        if isinstance(campos, dict):
            txt += f"\n[{categoria.upper()}]\n"
            for k, v in campos.items():
                if v and str(v).strip():
                    txt += f"- {k}: {v}\n"
        else:
            if campos and str(campos).strip():
                txt += f"{categoria}: {campos}\n"
    return txt