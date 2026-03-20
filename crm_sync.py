import json
import os

from crm.zoho_modules import (
    fetch_leads,
    fetch_contacts,
    fetch_notes,
    fetch_events,
    fetch_calls,
    fetch_tasks,
    fetch_deals
)

os.makedirs("data", exist_ok=True)


def save(name, data):

    path = f"data/{name}.json"

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"{name} guardado")


def run():

    save("leads", fetch_leads())
    save("contacts", fetch_contacts())
    save("notes", fetch_notes())
    save("events", fetch_events())
    save("calls", fetch_calls())
    save("tasks", fetch_tasks())
    save("deals", fetch_deals())

    print("SYNC COMPLETO")


if __name__ == "__main__":
    run()