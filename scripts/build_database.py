import sqlite3
import json
from pathlib import Path

DATA_DIR = Path("data")
DB_PATH = DATA_DIR / "crm.db"

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# limpiar tablas
c.execute("DROP TABLE IF EXISTS leads")
c.execute("DROP TABLE IF EXISTS contacts")
c.execute("DROP TABLE IF EXISTS notes")

# tablas
c.execute("""
CREATE TABLE leads (
    id TEXT,
    name TEXT,
    company TEXT,
    email TEXT,
    phone TEXT,
    owner TEXT
)
""")

c.execute("""
CREATE TABLE contacts (
    id TEXT,
    name TEXT,
    email TEXT,
    phone TEXT,
    owner TEXT
)
""")

c.execute("""
CREATE TABLE notes (
    id TEXT,
    parent_id TEXT,
    note TEXT
)
""")

# -------------------------
# LEADS
# -------------------------

leads_file = DATA_DIR / "leads.json"

if leads_file.exists():

    with open(leads_file, "r", encoding="utf-8") as f:
        leads = json.load(f)

    for lead in leads:

        name = lead.get("Full_Name") or lead.get("Last_Name") or ""
        company = lead.get("Company") or ""
        email = lead.get("Email") or ""
        phone = lead.get("Phone") or ""
        owner = ""

        owner_data = lead.get("Owner")
        if isinstance(owner_data, dict):
            owner = owner_data.get("name", "")

        c.execute(
            "INSERT INTO leads VALUES (?,?,?,?,?,?)",
            (lead.get("id"), name, company, email, phone, owner)
        )

    print("Leads cargados")

# -------------------------
# CONTACTS
# -------------------------

contacts_file = DATA_DIR / "contacts.json"

if contacts_file.exists():

    with open(contacts_file, "r", encoding="utf-8") as f:
        contacts = json.load(f)

    for contact in contacts:

        name = contact.get("Full_Name") or ""
        email = contact.get("Email") or ""
        phone = contact.get("Phone") or ""
        owner = ""

        owner_data = contact.get("Owner")
        if isinstance(owner_data, dict):
            owner = owner_data.get("name", "")

        c.execute(
            "INSERT INTO contacts VALUES (?,?,?,?,?)",
            (contact.get("id"), name, email, phone, owner)
        )

    print("Contacts cargados")

# -------------------------
# NOTES
# -------------------------

notes_file = DATA_DIR / "notes.json"

if notes_file.exists():

    with open(notes_file, "r", encoding="utf-8") as f:
        notes = json.load(f)

    for note in notes:

        text = note.get("Note_Content") or ""

        parent = ""
        parent_data = note.get("Parent_Id")

        if isinstance(parent_data, dict):
            parent = parent_data.get("id", "")

        c.execute(
            "INSERT INTO notes VALUES (?,?,?)",
            (note.get("id"), parent, text)
        )

    print("Notes cargadas")

conn.commit()
conn.close()

print("\nCRM cargado correctamente en SQLite")