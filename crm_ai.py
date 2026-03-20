import sqlite3
import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

DB = "data/crm.db"


def get_client_notes(name):

    conn = sqlite3.connect(DB)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT notes.content
    FROM notes
    JOIN accounts ON accounts.id = notes.parent_id
    WHERE accounts.name LIKE ?
    """, ("%" + name + "%",))

    rows = cursor.fetchall()

    conn.close()

    return [r[0] for r in rows]


def analyze_client(name):

    notes = get_client_notes(name)

    if not notes:
        print("No hay notas para este cliente")
        return

    text = "\n".join(notes[:30])

    prompt = f"""
Eres un director comercial experto.

Analiza estas notas de CRM sobre un cliente y responde:

1 resumen del cliente
2 estado del cliente (frio tibio caliente)
3 oportunidades detectadas
4 riesgos
5 plan de accion

Notas:

{text}
"""

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    print("\nANALISIS IA\n")
    print(response.choices[0].message.content)


if __name__ == "__main__":

    name = input("Cliente: ")

    analyze_client(name)