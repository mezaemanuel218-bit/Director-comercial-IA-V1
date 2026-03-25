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
        desktop_db = Path(r"C:\Users\sopor\OneDrive\Datos adjuntos\Desktop\Director Comercial IA\data\warehouse.db")
        db_path = str(desktop_db) if desktop_db.exists() else None
        cls.service = SalesAssistantService(db_path=db_path)
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

    def test_sales_email_draft_query_should_be_detected(self) -> None:
        intent = classify_question(
            "dame un correo electronico para mandarle a movimex, usa todo lo que sabes para tratar de lograr una venta"
        )
        self.assertTrue(intent.asks_for_sales_draft)

    def test_sales_email_draft_formatter_uses_context(self) -> None:
        draft = self.service._sales_draft(
            "dame un correo electronico para mandarle a movimex",
            {
                "entity_term": "movimex",
                "latest_row": {"company_name": "Movimex", "giro": "Logistica", "unit_count": 25, "unit_type": "vehiculos"},
                "contacts": [{"label": "Dena Salinas", "email": "dena.salinas@movimex.com", "phone": "662 214 2253"}],
                "recent_notes": [{"content_text": "Hubo visita y quieren demo. Actualmente usan gps y buscan seguimiento por whatsapp."}],
                "document_chunks": [{"file_name": "brochure.pdf", "content": "Soluciones de rastreo y monitoreo."}],
                "insights": {"signals": ["Hay actividad comercial real y conversacion activa en notas recientes."], "next_step": "Buscar cierre de fecha para demo o validacion operativa."},
            },
        )
        answer = self.service._format_sales_draft(draft).lower()
        self.assertIn("borrador de correo", answer)
        self.assertIn("movimex", answer)
        self.assertIn("asunto:", answer)
        self.assertIn("dena.salinas@movimex.com", answer)

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

    def test_action_plan_query_should_be_detected(self) -> None:
        intent = classify_question("analiza mis notas y arma un plan para hoy")
        self.assertTrue(intent.asks_for_action_plan)

    def test_analyze_my_notes_plan_keeps_owner_scope_not_random_entity(self) -> None:
        response = self.service.answer_question("analiza mis notas y arma un plan para hoy", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("plan comercial de hoy para", answer)
        self.assertIn("jesus emmanuel meza", answer)
        self.assertNotIn("trujillo fletes", answer)

    def test_sales_material_query_should_be_detected(self) -> None:
        intent = classify_question("hazme argumentos de venta para una llamada con cafenio")
        self.assertTrue(intent.asks_for_sales_material)

    def test_compound_question_keeps_entity_context(self) -> None:
        parts = self.service._contextualize_subquestions(
            "dame resumen de movimex y luego redactame un correo",
            ["dame resumen de movimex", "redactame un correo"],
        )
        self.assertEqual(parts[1], "redactame un correo para movimex")

    def test_question_mark_compound_keeps_entity_context(self) -> None:
        parts = self.service._contextualize_subquestions(
            "que notas hay de jibo? cuantos clientes o prospectos estan registrados a ese nombre?",
            ["que notas hay de jibo", "cuantos clientes o prospectos estan registrados a ese nombre"],
        )
        self.assertIn("jibo", parts[1].lower())

    def test_best_clients_for_owner_returns_ranked_shortlist(self) -> None:
        response = self.service.answer_question("los mejores tres clientes de Eduardo y por que")
        answer = response.answer.lower()
        self.assertIn("top 3", answer)
        self.assertIn("eduardo valdez", answer)
        self.assertNotIn("clientes y prospectos de eduardo valdez", answer)

    def test_count_my_clients_uses_logged_owner_scope(self) -> None:
        response = self.service.answer_question("cuantos clientes tengo registrados o dados de alta?", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("jesus emmanuel meza", answer)
        self.assertIn("clientes o prospectos visibles", answer)
        self.assertNotIn("ventura rios", answer)

    def test_count_for_entity_returns_entity_summary(self) -> None:
        response = self.service.answer_question("cuantos clientes o prospectos estan registrados con jibe?")
        answer = response.answer.lower()
        self.assertIn("registros detectados", answer)
        self.assertIn("jibe", answer)
        self.assertNotIn("ventura rios", answer)

    def test_fuzzy_jibo_does_not_hallucinate_other_account(self) -> None:
        response = self.service.answer_question("que me dices de jibo? que plan de accion me recomiendas para el?")
        answer = response.answer.lower()
        self.assertNotIn("la loma base para helados", answer)
        self.assertTrue("jibe" in answer or "no encontre informacion registrada exactamente" in answer)

    def test_typo_entity_returns_suggestions_instead_of_wrong_account(self) -> None:
        response = self.service.answer_question("que notas hay de jibo?")
        answer = response.answer.lower()
        self.assertNotIn("transportes morquecho", answer)
        self.assertTrue("lo mas cercano" in answer or "jibe" in answer)

    def test_precall_brief_query_should_be_treated_as_client_brief(self) -> None:
        response = self.service.answer_question("si entro a una llamada en 5 minutos con movimex, que debo tener claro")
        answer = response.answer.lower()
        self.assertIn("movimex", answer)
        self.assertNotIn("no encontre evidencia suficiente para responder", answer)

    def test_status_query_for_entity_should_not_need_exact_phrase(self) -> None:
        response = self.service.answer_question("Como vamos con JIBE?")
        answer = response.answer.lower()
        self.assertIn("jibe", answer)
        self.assertNotIn("no encontre informacion registrada exactamente", answer)

    def test_recent_activity_query_for_owner_uses_owner_scope(self) -> None:
        response = self.service.answer_question("que actividad esta realizando Eduardo Valdez en CRM")
        answer = response.answer.lower()
        self.assertIn("eduardo valdez", answer)
        self.assertNotIn("no encontre evidencia suficiente para responder", answer)

    def test_geotab_leads_owner_alias_with_underscore_is_resolved(self) -> None:
        response = self.service.answer_question("cuales son registros de clientes de geotab_leads")
        answer = response.answer.lower()
        self.assertIn("geotab", answer)
        self.assertNotIn("no encontre clientes asignados", answer)

    def test_geotab_owner_alias_without_underscore_is_resolved(self) -> None:
        response = self.service.answer_question("cuales son registros de clientes de geotab")
        answer = response.answer.lower()
        self.assertIn("geotab", answer)
        self.assertNotIn("no encontre clientes asignados", answer)

    def test_free_minutes_work_plan_should_not_jump_to_random_account(self) -> None:
        response = self.service.answer_question("tengo 30 min libres, hazme un plan de trabajo", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("plan comercial de hoy para", answer)
        self.assertNotIn("salci", answer)

    def test_owner_status_query_returns_owner_brief(self) -> None:
        response = self.service.answer_question("como va Eliot Hernandez comercialmente")
        answer = response.answer.lower()
        self.assertIn("eliot hernandez", answer)
        self.assertNotIn('no encontre informacion registrada exactamente para "como va eliot hernandez comercialmente"', answer)

    def test_owner_pending_today_query_returns_today_pending(self) -> None:
        response = self.service.answer_question("que pendientes tiene Ayuda consultoria hoy")
        answer = response.answer.lower()
        self.assertTrue("lo mas reciente visible" in answer or "pendiente" in answer or "compromisos de hoy" in answer)

    def test_decision_maker_question_uses_entity_context(self) -> None:
        response = self.service.answer_question("que decision maker ves en Condesa")
        answer = response.answer.lower()
        self.assertIn("condesa", answer)
        self.assertNotIn("no encontre evidencia suficiente para responder", answer)


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
