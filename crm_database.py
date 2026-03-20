import sqlite3
import json
import os

DB = "data/crm.db"


def create_database():

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS accounts(
        id TEXT PRIMARY KEY,
        name TEXT,
        phone TEXT,
        website TEXT,
        city TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS notes(
        id TEXT PRIMARY KEY,
        parent_id TEXT,
        content TEXT,
        created_time TEXT
    )
    """)

    conn.commit()
    conn.close()

    print("Base de datos creada")


def load_json(file):

    path = f"data/{file}.json"

    if not os.path.exists(path):
        return []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return data


def insert_accounts():

    accounts = load_json("accounts")

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    for a in accounts:

        c.execute("""
        INSERT OR REPLACE INTO accounts
        VALUES(?,?,?,?,?)
        """, (

            a.get("id"),
            a.get("Account_Name"),
            a.get("Phone"),
            a.get("Website"),
            a.get("Billing_City")

        ))

    conn.commit()
    conn.close()

    print("Accounts cargados")


def insert_notes():

    notes = load_json("notes")

    conn = sqlite3.connect(DB)
    c = conn.cursor()

    for n in notes:

        parent = n.get("Parent_Id")

        parent_id = None

        if isinstance(parent, dict):
            parent_id = parent.get("id")

        c.execute("""
        INSERT OR REPLACE INTO notes
        VALUES(?,?,?,?)
        """, (

            n.get("id"),
            parent_id,
            n.get("Note_Content"),
            n.get("Created_Time")

        ))

    conn.commit()
    conn.close()

    print("Notes cargadas")


def build_database():

    create_database()
    insert_accounts()
    insert_notes()

    print("CRM cargado en SQLite")


if __name__ == "__main__":
    build_database()