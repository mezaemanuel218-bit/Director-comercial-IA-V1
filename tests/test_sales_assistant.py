import unittest
from pathlib import Path

from assistant_core.auth import get_user
from assistant_core.history import ensure_history_schema, fetch_history, save_history
from assistant_core.query_intent import classify_question
from assistant_core.service import SalesAssistantService
import assistant_core.history as history_module


class SalesAssistantServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = SalesAssistantService()
        cls.emeza = get_user("emeza")

    def test_movimex_contact_query_detects_note_and_crm_emails(self) -> None:
        response = self.service.answer_question("dame solo los correos y nombres de movimex")
        answer = response.answer.lower()
        self.assertIn("dena", answer)
        self.assertIn("movimex", answer)
        self.assertIn("gps@movimex.mx", answer)

    def test_bare_client_name_returns_client_context(self) -> None:
        response = self.service.answer_question("movimex")
        answer = response.answer.lower()
        self.assertIn("movimex", answer)
        self.assertNotIn("no hay informacion registrada", answer)

    def test_client_weekly_kpi_should_not_be_treated_as_owner_kpi(self) -> None:
        response = self.service.answer_question("kpi movimex de la semana")
        answer = response.answer.lower()
        self.assertIn("movimex", answer)
        self.assertIn("indicadores comerciales", answer)

    def test_today_call_list_is_scoped_to_logged_seller(self) -> None:
        response = self.service.answer_question("a quien debo llamar hoy y por que", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("rd security", answer)
        self.assertNotIn("tsm", answer)
        self.assertNotIn("mudanzas milenio", answer)

    def test_global_owner_load_lists_known_sellers(self) -> None:
        response = self.service.answer_question("quien tiene mas clientes asignados")
        answer = response.answer
        self.assertIn("Eduardo Valdez", answer)
        self.assertIn("Jesus Emmanuel Meza", answer)
        self.assertIn("Pablo Melin Dorador", answer)

    def test_entity_comparison_uses_existing_crm_evidence(self) -> None:
        response = self.service.answer_question("compara Hieleria Veracruz vs Servicios y Minas de Mexico Actus")
        answer = response.answer.lower()
        self.assertIn("hieleria veracruz", answer)
        self.assertIn("servicios y minas de mexico actus", answer)
        self.assertNotIn("no hay registros en zoho crm ni pdfs locales sobre hieleria veracruz", answer)

    def test_kpis_for_emmanuel_are_not_zero(self) -> None:
        response = self.service.answer_question("mis kpis", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("jesus emmanuel meza", answer)
        self.assertNotIn("interacciones: 0", answer)

    def test_weekly_kpis_for_self_scope_are_resolved(self) -> None:
        response = self.service.answer_question("kpi mio de la semana", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("jesus emmanuel meza", answer)
        self.assertIn("ultimos 7 dias", answer)

    def test_plural_differences_query_should_be_treated_as_comparison(self) -> None:
        intent = classify_question("diferencias entre emmanuel y pablo melin")
        self.assertTrue(intent.asks_for_comparison)

    def test_open_summary_query_should_be_treated_as_owner_brief(self) -> None:
        intent = classify_question("dime todo lo que debo saber de mis contactos o leads")
        self.assertTrue(intent.asks_for_owner_brief)

    def test_recommendation_call_query_should_be_treated_as_today_call_list(self) -> None:
        intent = classify_question("en base a mis notas a quien me recomiendas llamar hoy")
        self.assertTrue(intent.asks_for_today_call_list)

    def test_latest_note_query_should_be_detected(self) -> None:
        intent = classify_question("ultima nota agregada")
        self.assertTrue(intent.asks_for_latest_note)

    def test_owner_comparison_returns_metrics_for_both_sellers(self) -> None:
        response = self.service.answer_question("diferencias entre emmanuel y pablo melin")
        answer = response.answer.lower()
        self.assertIn("jesus emmanuel meza", answer)
        self.assertIn("pablo melin dorador", answer)
        self.assertIn("interacciones", answer)

    def test_owner_brief_lists_hot_and_cold_accounts(self) -> None:
        response = self.service.answer_question("dame clientes calientes y clientes frios", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("clientes calientes", answer)
        self.assertIn("clientes frios", answer)


class HistoryIsolationTests(unittest.TestCase):
    def test_history_can_be_filtered_by_username(self) -> None:
        original_db = history_module.WAREHOUSE_DB
        temp_dir = Path("data/test-history")
        temp_dir.mkdir(parents=True, exist_ok=True)
        history_db = temp_dir / "history.db"
        if history_db.exists():
            history_db.unlink()
        try:
            history_module.WAREHOUSE_DB = history_db
            ensure_history_schema()
            save_history("q1", "a1", "data", ["warehouse.db"], False, username="emeza")
            save_history("q2", "a2", "data", ["warehouse.db"], False, username="pmelin")

            emeza_items = fetch_history(username="emeza")
            pmelin_items = fetch_history(username="pmelin")
            all_items = fetch_history()

            self.assertEqual(len(emeza_items), 1)
            self.assertEqual(len(pmelin_items), 1)
            self.assertEqual(emeza_items[0]["username"], "emeza")
            self.assertEqual(pmelin_items[0]["username"], "pmelin")
            self.assertEqual(len(all_items), 2)
        finally:
            history_module.WAREHOUSE_DB = original_db
            if history_db.exists():
                history_db.unlink()


if __name__ == "__main__":
    unittest.main()
