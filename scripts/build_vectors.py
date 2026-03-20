import os
import sys
import pickle

# -------------------------------------------------
# ARREGLAR PATH DEL PROYECTO
# -------------------------------------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# IMPORTS
# -------------------------------------------------

from ia.pdf_loader import load_pdfs
from ia.embeddings import embed
from ia.vector_store import VectorStore

print("Cargando PDFs...")

pdfs = load_pdfs()

store = VectorStore()

for doc in pdfs:

    # dividir documentos grandes
    parts = [doc[i:i+1000] for i in range(0, len(doc), 1000)]

    for p in parts:

        print("embedding...")

        v = embed(p)

        store.add(p, v)

os.makedirs("data", exist_ok=True)

with open("data/vector_store.pkl", "wb") as f:
    pickle.dump(store, f)

print("Vectores guardados")