import unittest
import uuid
from pathlib import Path

import api.app as api_app
import assistant_core.documents as documents_module
import assistant_core.history as history_module
import assistant_core.reporting as reporting_module
import assistant_core.runtime_state as runtime_state_module
import assistant_core.warehouse as warehouse_module
from assistant_core.auth import get_user
from assistant_core.service import SalesAssistantService


class BootstrapStartupTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_api_warehouse = api_app.WAREHOUSE_DB
        self.original_api_bootstrap = api_app.BOOTSTRAP_WAREHOUSE_DB
        self.original_runtime_db = runtime_state_module.WAREHOUSE_DB
        self.original_history_db = history_module.WAREHOUSE_DB
        self.original_reporting_db = reporting_module.WAREHOUSE_DB
        self.original_documents_db = documents_module.WAREHOUSE_DB
        self.original_warehouse_db = warehouse_module.WAREHOUSE_DB
        self.original_indexed_documents_count = api_app.indexed_documents_count
        self.original_index_documents = api_app.index_documents

        self.temp_dir = Path("data/test-bootstrap-startup")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        suffix = uuid.uuid4().hex
        self.test_warehouse = self.temp_dir / f"warehouse-{suffix}.db"
        self.test_bootstrap = self.temp_dir / f"warehouse.snapshot-{suffix}.db"
        source_bootstrap = Path("bootstrap/warehouse.snapshot.db")
        self.assertTrue(source_bootstrap.exists(), "bootstrap snapshot should exist")
        self.test_bootstrap.write_bytes(source_bootstrap.read_bytes())

        api_app.WAREHOUSE_DB = self.test_warehouse
        api_app.BOOTSTRAP_WAREHOUSE_DB = self.test_bootstrap
        runtime_state_module.WAREHOUSE_DB = self.test_warehouse
        history_module.WAREHOUSE_DB = self.test_warehouse
        reporting_module.WAREHOUSE_DB = self.test_warehouse
        documents_module.WAREHOUSE_DB = self.test_warehouse
        warehouse_module.WAREHOUSE_DB = self.test_warehouse
        api_app.indexed_documents_count = lambda: 1
        api_app.index_documents = lambda: 0

    def tearDown(self) -> None:
        api_app.WAREHOUSE_DB = self.original_api_warehouse
        api_app.BOOTSTRAP_WAREHOUSE_DB = self.original_api_bootstrap
        runtime_state_module.WAREHOUSE_DB = self.original_runtime_db
        history_module.WAREHOUSE_DB = self.original_history_db
        reporting_module.WAREHOUSE_DB = self.original_reporting_db
        documents_module.WAREHOUSE_DB = self.original_documents_db
        warehouse_module.WAREHOUSE_DB = self.original_warehouse_db
        api_app.indexed_documents_count = self.original_indexed_documents_count
        api_app.index_documents = self.original_index_documents
        for path in (self.test_warehouse, self.test_bootstrap):
            try:
                if path.exists():
                    path.unlink()
            except PermissionError:
                pass

    def test_prepare_local_state_restores_bootstrap_snapshot(self) -> None:
        api_app.prepare_local_state()
        self.assertTrue(self.test_warehouse.exists())

        service = SalesAssistantService(db_path=str(self.test_warehouse))
        response = service.answer_question("kpi mio de la semana", user=get_user("emeza"))
        answer = response.answer.lower()
        self.assertIn("jesus emmanuel meza", answer)
        self.assertNotIn("interacciones totales: 0", answer)

    def test_bootstrap_snapshot_supports_client_brief(self) -> None:
        api_app.prepare_local_state()
        service = SalesAssistantService(db_path=str(self.test_warehouse))
        response = service.answer_question("dame todo lo que debo saber de Hieleria Veracruz", user=get_user("emeza"))
        answer = response.answer.lower()
        self.assertIn("hieleria veracruz", answer)
        self.assertNotIn("sin suficiente evidencia estructurada", answer)


if __name__ == "__main__":
    unittest.main()
