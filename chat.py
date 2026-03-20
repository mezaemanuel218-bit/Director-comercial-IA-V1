import sqlite3
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import logging

logging.getLogger("pypdf").setLevel(logging.ERROR)

# ============================
# CONFIG
# ============================

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB = "data/crm.db"
DOC_FOLDER = "doc"

# ============================
# DETECTAR EMPRESA
# ============================

def detectar_empresa(pregunta):

    prompt = f"""
Extrae SOLO el nombre de la empresa de la frase.

Frase:
{pregunta}

Si no hay empresa responde: NONE
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    empresa = r.choices[0].message.content.strip()

    if empresa.upper() == "NONE":
        return None

    return empresa


# ============================
# BUSCAR EN CRM (CORREGIDO)
# ============================

def buscar_empresa(empresa):

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    q = f"%{empresa.lower()}%"

    # LEADS (principal)
    c.execute("""
    SELECT id,name,company,email,phone,owner
    FROM leads
    WHERE LOWER(name) LIKE ?
       OR LOWER(company) LIKE ?
    """,(q,q))

    leads = c.fetchall()

    # CONTACTS (NO tiene company → corregido)
    c.execute("""
    SELECT id,name,email,phone,owner
    FROM contacts
    WHERE LOWER(name) LIKE ?
    """,(q,))

    contacts = c.fetchall()

    # NOTAS
    notas = []

    for l in leads:
        lead_id = l[0]

        c.execute("""
        SELECT note
        FROM notes
        WHERE parent_id = ?
        """,(lead_id,))

        notas += c.fetchall()

    conn.close()

    return leads,contacts,notas


# ============================
# EXTRAER INFO DE NOTAS
# ============================

def extraer_datos_notas(notas):

    texto = " ".join([n[0] for n in notas])

    correos = re.findall(r'[\w\.-]+@[\w\.-]+', texto)
    telefonos = re.findall(r'\d{3}[\s\-]?\d{3}[\s\-]?\d{4}', texto)

    # nombres simples (heurística básica)
    posibles_nombres = re.findall(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', texto)

    return list(set(correos)), list(set(telefonos)), list(set(posibles_nombres))


# ============================
# LEER PDF
# ============================

def leer_documentos(empresa):

    texto = ""

    if not os.path.exists(DOC_FOLDER):
        return texto

    for archivo in os.listdir(DOC_FOLDER):

        if archivo.lower().endswith(".pdf"):

            ruta = os.path.join(DOC_FOLDER,archivo)

            try:
                reader = PdfReader(ruta)

                for page in reader.pages:

                    try:
                        contenido = page.extract_text()

                        if contenido and empresa.lower() in contenido.lower():
                            texto += contenido + "\n"

                    except Exception:
                        continue

            except Exception:
                continue

    return texto

# ============================
# CONTEXTO BIEN ARMADO
# ============================

def preparar_contexto(empresa,leads,contacts,notas,documentos):

    contexto = f"EMPRESA: {empresa}\n\n"

    # LEADS
    for l in leads:
        contexto += f"""
Lead:
Nombre: {l[1]}
Empresa: {l[2]}
Email: {l[3]}
Teléfono: {l[4]}
Vendedor: {l[5]}
"""

    # CONTACTOS
    for c in contacts:
        contexto += f"""
Contacto adicional:
Nombre: {c[1]}
Email: {c[2]}
Teléfono: {c[3]}
Vendedor: {c[4]}
"""

    # NOTAS (RAW)
    if notas:
        contexto += "\nNOTAS:\n"
        for n in notas[:20]:
            contexto += f"- {n[0]}\n"

    # EXTRAER INTELIGENCIA DE NOTAS
    correos,telefonos,nombres = extraer_datos_notas(notas)

    contexto += f"""
DATOS EXTRAIDOS DE NOTAS:
Correos detectados: {correos}
Teléfonos detectados: {telefonos}
Nombres detectados: {nombres}
"""

    # DOCS
    if documentos:
        contexto += f"\nDOCUMENTOS:\n{documentos[:3000]}"

    return contexto


# ============================
# GPT FINAL (CLAVE)
# ============================

def analizar(pregunta,contexto):

    tipo = tipo_pregunta(pregunta)

    if tipo == "simple":
        instrucciones = """
Responde SOLO con:

Nombres de contactos
Vendedor

NO agregues nada más.
"""

    elif tipo == "estrategia":
        instrucciones = """
Responde SOLO con estrategia comercial explicada y justificada.
"""

    else:
        instrucciones = """
Responde completo en formato:

Resumen del cliente
Contactos detectados
Canales de comunicación
Responsable comercial
Información clave
Estado
Estrategia
"""

    prompt = f"""
Eres un director comercial experto.

REGLAS:
- Usa SOLO el contexto
- NO inventes
- Sé claro y directo

CONTEXTO:
{contexto}

PREGUNTA:
{pregunta}

INSTRUCCIONES:
{instrucciones}
"""

    r = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return r.choices[0].message.content

# ============================

def tipo_pregunta(pregunta):

    p = pregunta.lower()

    if "solo" in p or "nombres" in p:
        return "simple"

    if "estrategia" in p or "vender" in p:
        return "estrategia"

    if "resumen" in p:
        return "resumen"

    return "completo"

# CHAT
# ============================

print("\nDirector Comercial IA listo\n")

while True:

    pregunta = input("Tú: ")

    if pregunta.lower() in ["salir","exit","quit"]:
        break

    empresa = detectar_empresa(pregunta)

    if not empresa:
        print("\nIA: Dime la empresa que quieres analizar.")
        continue

    leads,contacts,notas = buscar_empresa(empresa)

    documentos = leer_documentos(empresa)

    contexto = preparar_contexto(empresa,leads,contacts,notas,documentos)

    respuesta = analizar(pregunta,contexto)

    print("\nIA:\n",respuesta)