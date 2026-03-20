import sqlite3

DB = "data/crm.db"


def buscar_empresa(nombre):

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    q = f"%{nombre.lower()}%"

    # buscar en leads (empresa está en name)
    c.execute("""
        SELECT name,email,phone,owner
        FROM leads
        WHERE LOWER(name) LIKE ?
    """, (q,))
    leads = c.fetchall()

    # buscar en contactos
    c.execute("""
        SELECT name,email,phone,owner
        FROM contacts
        WHERE LOWER(name) LIKE ?
    """, (q,))
    contacts = c.fetchall()

    conn.close()

    return leads, contacts


def formatear_respuesta(nombre, leads, contacts):

    if not leads and not contacts:
        return f"No encontré registros para '{nombre}' en tu CRM."

    texto = "\nINFORMACIÓN ENCONTRADA EN CRM\n\n"

    if leads:
        texto += "LEADS:\n"
        for l in leads:
            texto += f"""
Empresa/Lead: {l[0]}
Email: {l[1]}
Teléfono: {l[2]}
Vendedor: {l[3]}
"""
    if contacts:
        texto += "\nCONTACTOS:\n"
        for c in contacts:
            texto += f"""
Nombre/Empresa: {c[0]}
Email: {c[1]}
Teléfono: {c[2]}
Vendedor: {c[3]}
"""

    return texto


def analizar_empresa(nombre):

    leads, contacts = buscar_empresa(nombre)

    return formatear_respuesta(nombre, leads, contacts)