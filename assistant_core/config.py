from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOC_DIR = PROJECT_ROOT / "doc"
WAREHOUSE_DB = DATA_DIR / "warehouse.db"
BOOTSTRAP_WAREHOUSE_DB = PROJECT_ROOT / "bootstrap" / "warehouse.snapshot.db"

RAW_MODULE_FILES = {
    "leads": DATA_DIR / "leads.json",
    "contacts": DATA_DIR / "contacts.json",
    "notes": DATA_DIR / "notes.json",
    "calls": DATA_DIR / "calls.json",
    "tasks": DATA_DIR / "tasks.json",
    "events": DATA_DIR / "events.json",
}

