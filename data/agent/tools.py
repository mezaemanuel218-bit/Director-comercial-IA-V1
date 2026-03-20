import sqlite3
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB = "data/crm.db"


def sql_query(query):

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute(query)

    rows = cursor.fetchall()

    conn.close()

    return rows


def search_client(name):

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT id, name, phone, website, city
    FROM accounts
    WHERE name LIKE ?
    """, ("%" + name + "%",))

    result = cursor.fetchone()

    conn.close()

    return result


def client_notes(client_id):

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT content, created_time
    FROM notes
    WHERE parent_id = ?
    ORDER BY created_time DESC
    """, (client_id,))

    rows = cursor.fetchall()

    conn.close()

    return rows


def analyze_client(name):

    client_data = search_client(name)

    if not client_data:
        return "Cliente no encontrado"

    client_id, name, phone, website, city = client_data

    notes = client_notes(client_id)

    text = ""

    for n in notes[:30]:

        content, date = n

        text += f"\nFecha:{date}\n{content}\n"

    prompt = f"""
Analiza este cliente usando las notas de CRM.

Cliente: {name}

Notas:

{text}

Responde:

1 resumen del cliente
2 estado comercial (frio tibio caliente)
3 oportunidades
4 riesgos
5 siguiente accion recomendada
"""

    r = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    return r.choices[0].message.content


def kpi_contacts_last_week():

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT COUNT(*)
    FROM contacts
    WHERE created_time >= date('now','-7 day')
    """)

    result = cursor.fetchone()[0]

    conn.close()

    return result


def owners_stats():

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT owner, COUNT(*)
    FROM contacts
    GROUP BY owner
    """)

    rows = cursor.fetchall()

    conn.close()

    return rows