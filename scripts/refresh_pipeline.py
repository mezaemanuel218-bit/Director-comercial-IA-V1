import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from assistant_core.documents import index_documents
from assistant_core.warehouse import build_warehouse
from scripts.sync_zoho import run as sync_zoho


def main() -> None:
    print("1. Sincronizando Zoho...")
    sync_zoho()

    print("\n2. Construyendo warehouse...")
    stats = build_warehouse()
    print(
        f"Warehouse listo | leads={stats.leads} contacts={stats.contacts} "
        f"notes={stats.notes} calls={stats.calls} tasks={stats.tasks} "
        f"events={stats.events} interactions={stats.interactions}"
    )

    print("\n3. Indexando documentos...")
    total_docs = index_documents()
    print(f"Documentos indexados: {total_docs}")

    print("\nPIPELINE COMPLETO")


if __name__ == "__main__":
    main()
