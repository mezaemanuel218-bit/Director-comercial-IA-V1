import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from assistant_core.documents import index_documents


def main() -> None:
    total = index_documents()
    print(f"Documentos indexados: {total}")


if __name__ == "__main__":
    main()

