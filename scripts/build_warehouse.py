import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from assistant_core.warehouse import build_warehouse


def main() -> None:
    stats = build_warehouse()
    print("Warehouse construido")
    print(f"Leads: {stats.leads}")
    print(f"Contacts: {stats.contacts}")
    print(f"Notes: {stats.notes}")
    print(f"Calls: {stats.calls}")
    print(f"Tasks: {stats.tasks}")
    print(f"Events: {stats.events}")
    print(f"Interactions: {stats.interactions}")


if __name__ == "__main__":
    main()

