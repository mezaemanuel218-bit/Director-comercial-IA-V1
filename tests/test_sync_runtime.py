import unittest
from pathlib import Path

import assistant_core.runtime_state as runtime_state_module
import assistant_core.sync_runtime as sync_runtime_module
from assistant_core.runtime_state import ensure_runtime_schema, set_runtime_value
from assistant_core.sync_runtime import acquire_sync_lock, get_sync_snapshot, refresh_is_stale, release_sync_lock


class SyncRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original_runtime_db = runtime_state_module.WAREHOUSE_DB
        self.original_sync_db = sync_runtime_module.get_runtime_value.__globals__["WAREHOUSE_DB"]
        temp_dir = Path("data/test-sync-runtime")
        temp_dir.mkdir(parents=True, exist_ok=True)
        self.test_db = temp_dir / "runtime.db"
        if self.test_db.exists():
            self.test_db.unlink()
        runtime_state_module.WAREHOUSE_DB = self.test_db
        sync_runtime_module.get_runtime_value.__globals__["WAREHOUSE_DB"] = self.test_db
        sync_runtime_module.set_runtime_value.__globals__["WAREHOUSE_DB"] = self.test_db
        ensure_runtime_schema()

    def tearDown(self) -> None:
        runtime_state_module.WAREHOUSE_DB = self.original_runtime_db
        sync_runtime_module.get_runtime_value.__globals__["WAREHOUSE_DB"] = self.original_sync_db
        sync_runtime_module.set_runtime_value.__globals__["WAREHOUSE_DB"] = self.original_sync_db
        if self.test_db.exists():
            self.test_db.unlink()

    def test_sync_lock_lifecycle(self) -> None:
        self.assertTrue(acquire_sync_lock("manual:emeza"))
        snapshot = get_sync_snapshot()
        self.assertTrue(snapshot.sync_in_progress)
        self.assertEqual(snapshot.sync_requested_by, "manual:emeza")

        self.assertFalse(acquire_sync_lock("manual:pmelin"))
        release_sync_lock()
        snapshot = get_sync_snapshot()
        self.assertFalse(snapshot.sync_in_progress)

    def test_refresh_is_stale_without_successful_refresh(self) -> None:
        self.assertTrue(refresh_is_stale(60))
        set_runtime_value("last_refresh", "manual_refresh")
        set_runtime_value("last_refresh_status", "ok")
        self.assertFalse(refresh_is_stale(60))


if __name__ == "__main__":
    unittest.main()
