import json
import os
import sys

# agregar carpeta raíz del proyecto al path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from crm.zoho_modules import (
    fetch_leads,
    fetch_contacts,
    fetch_notes,
    fetch_events,
    fetch_calls,
    fetch_tasks
)

DATA_DIR = os.path.join(ROOT_DIR, "data")
SYNC_MODULES = {
    "leads": fetch_leads,
    "contacts": fetch_contacts,
    "notes": fetch_notes,
    "events": fetch_events,
    "calls": fetch_calls,
    "tasks": fetch_tasks,
}


def save(name, data):

    os.makedirs(DATA_DIR, exist_ok=True)

    path = os.path.join(DATA_DIR, f"{name}.json")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"{name}.json guardado ({len(data)} registros)")


def run():

    print("\nSINCRONIZANDO ZOHO CRM...\n")

    for name, fetcher in SYNC_MODULES.items():
        save(name, fetcher())

    print("\nSYNC COMPLETO")


if __name__ == "__main__":
    run()
