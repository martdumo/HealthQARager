import streamlit as st
import requests, json, sys, subprocess
from pathlib import Path
import time
from patients_manager import guardar_paciente, cargar_pacientes, paciente_a_texto

ROOT_DIR = Path(__file__).resolve().parent.parent

def llm_call_with_antiloop(messages, json_mode=False):
    url = "http://127.0.0.1:1234/v1/chat/completions"
    for timeout, temp in [(45, 0.2), (120, 0.4), (None, 0.7)]:
        try:
            payload = {"model": "local-model", "messages": messages, "temperature": temp, "stream": False}
            if json_mode: payload["response_format"] = {"type": "json_object"}
            res = requests.post(url, json=payload, timeout=timeout)
            if res.status_code == 200:
                txt = res.json()['choices'][0]['message']['content']
                return json.loads(txt) if json_mode else txt
        except Exception: time.sleep(1)
    return {"error": "FAIL"} if json_mode else "ERROR: Timeout"

def run_search_subprocess(db_name, query, k=200):
    cmd = f'python "{ROOT_DIR}/src/step4_search.py" "{db_name}" "{query}" {k}'
    process = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    try: return json.loads(process.stdout)
    except: return []

def run_process_live(command):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, shell=True, bufsize=1)
    log_placeholder = st.empty()
    full_log = ""
    for line in iter(process.stdout.readline, ''):
        full_log += line
        log_placeholder.markdown(f"<div class='log-box'>{full_log[-2000:]}</div>", unsafe_allow_html=True)
    process.wait()

def inject_brutalist_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&display=swap');
    :root { --bg-color: #050505; --accent-color: #00FF41; --text-color: #E0E0E0; --border: 2px solid #00FF41; }
    .stApp { background-color: var(--bg-color); color: var(--text-color); font-family: 'JetBrains Mono', monospace; }
    h1, h2, h3 { color: var(--accent-color) !important; letter-spacing: 2px; }
    .stTextArea textarea, .stTextInput input, .stSelectbox div[data-baseweb="select"] { background-color: #111 !important; color: var(--accent-color) !important; border: var(--border) !important; border-radius: 0px !important; }
    .stButton>button { width: 100%; background: transparent !important; color: var(--accent-color) !important; border: var(--border) !important; border-radius: 0px !important; font-weight: bold; }
    .stButton>button:hover { background: var(--accent-color) !important; color: black !important; box-shadow: 0 0 15px var(--accent-color); }
    .result-box { border-left: 5px solid var(--accent-color); padding: 15px; margin-bottom: 20px; background: #111; font-size: 0.9em; }
    .log-box { background: #000; color: #00FF41; padding: 10px; height: 300px; overflow-y: auto; font-family: monospace;}
    #MainMenu, footer, header { visibility: hidden; }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="NEXUS_OS | EHR & RAG", layout="wide")
    inject_brutalist_css()

    if 'hits' not in st.session_state: st.session_state.hits = []
    if 'report' not in st.session_state: st.session_state.report = ""

    st.sidebar.title("NEXUS_OS RESEARCH")
    menu = st.sidebar.radio("Navegación:", ["1. PACIENTES (EHR)", "2. INDEXACIÓN", "3. BÚSQUEDA Y ANÁLISIS"])

    # ================== MÓDULO 1: PACIENTES ==================
    if menu == "1. PACIENTES (EHR)":
        st.title("SISTEMA DE HISTORIAS CLÍNICAS (PoC)")
        with st.form("paciente_form"):
            t1, t2, t3, t4 = st.tabs(["1. Identificación", "2. Motivo", "3. Antecedentes", "4. Estilo de Vida"])
            with t1:
                colA, colB = st.columns(2)
                p_nombre = colA.text_input("Nombre y Apellido:")
                p_dni = colB.text_input("DNI / Pasaporte:")
                p_edad = colA.text_input("Edad:")
                p_sexo = colB.selectbox("Sexo / Género:", ["", "Masculino", "Femenino", "Otro"])
            with t2:
                p_motivo = st.text_area("Motivo de Consulta Principal:")
                p_sintomas = st.text_area("Síntomas Asociados y Evolución:")
            with t3:
                c1, c2 = st.columns(2)
                p_ht = c1.checkbox("Hipertensión")
                p_db = c1.checkbox("Diabetes")
                p_alergias = st.text_input("Alergias (Medicamentos/Alimentos):")
                p_meds = st.text_area("Medicación Habitual:")
            with t4:
                p_fuma = st.selectbox("Tabaco:", ["", "No", "Sí"])
                p_mental = st.text_input("Salud Mental:")
                
            if st.form_submit_button("GUARDAR EHR") and p_nombre:
                datos = {"Nombre": p_nombre, "DNI": p_dni, "Edad": p_edad, "Sexo": p_sexo, "Motivo": p_motivo, "Síntomas": p_sintomas, "Alergias": p_alergias, "Medicación": p_meds}
                guardar_paciente(datos)
                st.success("Historia clínica guardada.")

    # ================== MÓDULO 2: INDEXACIÓN ==================
    elif menu == "2. INDEXACIÓN":
        st.title("PHASE 2: FAST GPU INDEXER")
        db_base = ROOT_DIR / "data" / "04_db"
        dbs = [d.name for d in db_base.iterdir() if d.is_dir()] if db_base.exists() else []
        db_name = st.text_input("NOMBRE DB:") if st.radio("OPERACIÓN:", ["Nueva", "Existente"]) == "Nueva" else st.selectbox("DB:", dbs)
        if st.button("CONSTRUIR ÍNDICE (BATCH ENCODING)") and db_name:
            run_process_live(f'python "{ROOT_DIR}/src/step3_indexer.py" {db_name}')

    # ================== MÓDULO 3: BÚSQUEDA Y ANÁLISIS ==================
    elif menu == "3. BÚSQUEDA Y ANÁLISIS":
        st.title("PHASE 3: MULTI-AGENT REFLEXIVE PIPELINE")
        db_base = ROOT_DIR / "data" / "04_db"
        dbs = [d.name for d in db_base.iterdir() if d.is_dir()] if db_base.exists() else []
        if not dbs: return st.warning("Sin bases vectoriales.")
        
        selected_db = st.sidebar.selectbox("TARGET DB:", dbs)
        pacientes = cargar_pacientes()
        opciones_pacientes = {"Nadie (Consulta General)": None}
        for p in pacientes: opciones_pacientes[f"{p['Nombre']} (DNI: {p['DNI']})"] = p
        paciente_seleccionado = st.sidebar.selectbox("EHR A INYECTAR:", list(opciones_pacientes.keys()))
        datos_paciente = opciones_pacientes[paciente_seleccionado]
        
        if st.sidebar.button("LIMPIAR"):
            st.session_state.hits = []
            st.session_state.report = ""
            st.rerun()

        col1, col2 = st.columns([2, 3])
        with col1:
            query = st.text_area("SÍNTOMAS / EVENTO EN EVALUACIÓN:")
            execute = st.button("INICIAR PROTOCOLO DE EXTRACCIÓN")
            
        if execute and query:
            st.session_state.hits = []
            st.session_state.report = ""
            
            with st.status("Iniciando Pipeline Reflexivo de Análisis...") as status:
                
                # 1. TRADUCCIÓN SILENCIOSA
                status.update(label="Traduciendo consulta para Motor Vectorial...")
                en_query = llm_call_with_antiloop([{"role": "user", "content": f"Translate to English. Only output the translation: {query}"}])

                # 2. BÚSQUEDA VECTORIAL
                status.update(label="Búsqueda Densa Multibase (Recall 200)...")
                all_hits = run_search_subprocess(selected_db, en_query, k=200)
                buckets = {"HealthQA": [], "Diseases": [], "Symptoms": [], "Dictionary": []}
                for h in all_hits:
                    f = h.get("fuente", "")
                    if f in buckets and len(buckets[f]) < 20: buckets[f].append(h)
                st.session_state.hits = all_hits[:30]
                
                # 3. MAP STAGE (Extracción)
                partial_reports = {}
                map_prompt = "CAVEMAN MODE. Extract ONLY ultra-relevant medical data related to the query. Max 1000 tokens. If no useful data, reply 'NO INFORMATION'."
                for fuente, hits_fuente in buckets.items():
                    if hits_fuente:
                        status.update(label=f"Agente Extractor procesando DB: {fuente}...")
                        contexto = "\n".join([h['texto'] for h in hits_fuente])
                        res = llm_call_with_antiloop([{"role": "system", "content": map_prompt}, {"role": "user", "content": f"DATA:\n{contexto[:20000]}\nQUERY: {en_query}"}])
                        if res and "NO INFORMATION" not in res.upper(): partial_reports[fuente] = res

                historia_clinica_txt = paciente_a_texto(datos_paciente)
                combined_context = "\n".join([f"--- BIBLIOGRAFÍA: {f} ---\n{r}" for f, r in partial_reports.items()])

                # 4. ORQUESTADOR 1.0 (Borrador Inicial)
                status.update(label="Orquestador 1.0: Generando Borrador Analítico Inicial...")
                draft_prompt = "Eres un Orquestador. Redacta en ESPAÑOL un borrador evaluando posibles diagnósticos, tratamientos y estudios sugeridos basados estrictamente en la bibliografía recuperada y los antecedentes de la historia clínica."
                borrador = llm_call_with_antiloop([{"role": "system", "content": draft_prompt}, {"role": "user", "content": f"EHR:\n{historia_clinica_txt}\n\nBIBLIOGRAFÍA:\n{combined_context}\n\nEVENTO: {query}"}])

                # 5. AGENTE AUDITOR 1: LOGICA DIAGNÓSTICA (Red Teaming)
                status.update(label="Agente Auditor 1: Cuestionando Hipótesis Diagnóstica...")
                critique1_prompt = """Eres un Auditor Médico Senior. Tu función es analizar de forma lógica y escéptica el BORRADOR proporcionado, contrastándolo con el EHR.
Marco lógico de revisión:
1. SESGO DE ANCLAJE: ¿El borrador se adelantó a una conclusión sin contemplar diagnósticos diferenciales más graves (Red Flags)?
2. CORRELACIÓN SINTOMÁTICA: Evalúa si los síntomas descritos justifican patologías alternativas no mencionadas.
Apunta únicamente a fallos lógicos en el diagnóstico. Redacta tus correcciones en ESPAÑOL."""
                critica_diag = llm_call_with_antiloop([{"role": "system", "content": critique1_prompt}, {"role": "user", "content": f"EHR:\n{historia_clinica_txt}\n\nBORRADOR A AUDITAR:\n{borrador}"}])

                # 6. AGENTE AUDITOR 2: SEGURIDAD, FARMACO E IATROGENIA
                status.update(label="Agente Auditor 2: Validación Lógica de Seguridad y Estudios...")
                critique2_prompt = """Eres un Auditor de Seguridad del Paciente (Farmacovigilancia y Prevención de Iatrogenia). Revisa lógicamente los tratamientos y estudios propuestos en el BORRADOR, cruzándolos de forma estricta con la Historia Clínica (EHR).
Aplica el siguiente marco lógico de validación deductiva:
1. INTERACCIONES: ¿Hay un choque directo entre las intervenciones sugeridas y las alergias o medicación actual del paciente descritas en el EHR?
2. RIESGOS EN ESTUDIOS DIAGNÓSTICOS: Evalúa la idoneidad de los estudios sugeridos (laboratorios, imágenes, procedimientos). ¿El paciente posee comorbilidades (fallas en órganos, embarazo, edad avanzada) que contraindiquen la técnica sugerida? Si un estudio propone un riesgo innecesario, sugiere un abordaje más seguro o advierte el riesgo.
3. AGRAVAMIENTO: ¿Alguna de las terapias recomendadas podría empeorar las condiciones crónicas subyacentes del EHR?
Redacta únicamente las alertas de seguridad y contraindicaciones detectadas en ESPAÑOL."""
                critica_trat = llm_call_with_antiloop([{"role": "system", "content": critique2_prompt}, {"role": "user", "content": f"EHR:\n{historia_clinica_txt}\n\nBORRADOR A AUDITAR:\n{borrador}"}])

                # 7. META-ORQUESTADOR FINAL
                status.update(label="Meta-Orquestador: Consolidando Síntesis Controlada...")
                meta_prompt = """Eres el Meta-Orquestador del sistema académico NEXUS.
Tu tarea es tomar el BORRADOR INICIAL y aplicar las estrictas correcciones lógicas de las DOS AUDITORÍAS para generar la salida final en ESPAÑOL.
REGLA 1 (BLINDAJE LEGAL OBLIGATORIO): Inicia tu respuesta exactamente con este texto: "⚠️ **ADVERTENCIA MÉDICA Y LEGAL:** El presente documento es una Síntesis Bibliográfica automatizada generada como Prueba de Concepto (PoC). NO ES UN DICTAMEN MÉDICO NI PRESCRIPCIÓN. La validación profesional humana es indispensable."
REGLA 2: Integra las correcciones lógicas. Si los auditores detectaron riesgos (interacciones, estudios contraindicados, diagnósticos omitidos), incorpóralos en MAYÚSCULAS en una sección llamada "ADVERTENCIAS CRÍTICAS DE AUDITORÍA (SEGURIDAD DEL PACIENTE)".
REGLA 3: Mantén la estructura clara: Análisis del Paciente, Posibles Patologías, Advertencias de Auditoría, y Protocolos/Estudios sugeridos basados en bibliografía."""

                prompt_final = f"BORRADOR INICIAL:\n{borrador}\n\nAUDITORÍA LÓGICA 1 (DIAGNÓSTICO):\n{critica_diag}\n\nAUDITORÍA LÓGICA 2 (SEGURIDAD Y ESTUDIOS):\n{critica_trat}"
                st.session_state.report = llm_call_with_antiloop([{"role": "system", "content": meta_prompt}, {"role": "user", "content": prompt_final}])
                
                status.update(label="Síntesis Bibliográfica Reflexiva Completada.", state="complete")
                st.rerun()

        if st.session_state.report:
            st.markdown("---")
            st.subheader("SÍNTESIS BIBLIOGRÁFICA GENERADA POR IA (NO VINCULANTE)")
            # Imprimimos el reporte final (que incluye la advertencia dictada por prompt).
            st.markdown(st.session_state.report)
            
            # Blindaje Hardcodeado en la UI (por si el LLM falla en seguir la regla)
            st.warning("⚖️ **DISCLAIMER LEGAL:** Este software no reemplaza el juicio clínico de un médico matriculado. Uso exclusivo para investigación de arquitecturas de IA.")
            
            st.markdown("---")
            with st.expander("VER MATERIAL BIBLIOGRÁFICO EXTRAÍDO (Evidencia Base)"):
                for i, h in enumerate(st.session_state.hits):
                    st.markdown(f"**[DB: {h.get('fuente')} | MATCH: {round(h['score']*100, 1)}%]** - ID: {h['chunk_id']}")
                    st.caption(h['texto'])

if __name__ == "__main__":
    main()