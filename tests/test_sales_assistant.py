import unittest
from pathlib import Path
from types import SimpleNamespace

from assistant_core.auth import get_user
from assistant_core.history import ensure_history_schema, fetch_feedback, fetch_feedback_memory, fetch_history, save_feedback, save_history
from assistant_core.query_intent import classify_question
from assistant_core.service import SalesAssistantService
import assistant_core.history as history_module


class SalesAssistantServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        desktop_db = Path(r"C:\Users\sopor\OneDrive\Datos adjuntos\Desktop\Director Comercial IA\data\warehouse.db")
        db_path = str(desktop_db) if desktop_db.exists() else None
        cls.service = SalesAssistantService(db_path=db_path)
        cls.service.client = None
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

    def test_contact_directory_query_keeps_directory_format(self) -> None:
        response = self.service.answer_question("dame nombres, telefonos y correos de contacto de movimex")
        answer = response.answer.lower()
        self.assertIn("contactos de movimex disponibles", answer)
        self.assertIn("dena salinas", answer)
        self.assertNotIn("asunto:", answer)

    def test_last_contact_query_returns_interactions_not_contact_directory(self) -> None:
        response = self.service.answer_question("ultimo contacto de movimex")
        answer = response.answer.lower()
        self.assertIn("movimex | note |", answer)
        self.assertNotIn("contactos de movimex disponibles", answer)

    def test_yesterday_contacts_query_is_not_treated_as_entity_lookup(self) -> None:
        response = self.service.answer_question("a quien le hable ayer", user=self.emeza)
        answer = response.answer.lower()
        self.assertNotIn("lo mas cercano que si veo en crm es", answer)

    def test_global_weekly_kpis_are_not_treated_as_entity_lookup(self) -> None:
        response = self.service.answer_question("kpi global de la semana de todos los vendedores", user=get_user("evaldez"))
        answer = response.answer.lower()
        self.assertIn("en corto:", answer)
        self.assertIn("panorama comercial del equipo flotimatics", answer)
        self.assertNotIn("lo mas cercano que si veo en crm es", answer)

    def test_executive_team_brief_uses_decision_style(self) -> None:
        response = self.service.answer_question("dame un resumen ejecutivo del equipo comercial", user=get_user("evaldez"))
        answer = response.answer.lower()
        self.assertIn("resumen ejecutivo del equipo", answer)
        self.assertIn("decision sugerida", answer)
        self.assertIn("siguiente paso recomendado", answer)

    def test_entity_comparison_uses_existing_crm_evidence(self) -> None:
        response = self.service.answer_question("compara Hieleria Veracruz vs Servicios y Minas de Mexico Actus")
        answer = response.answer.lower()
        self.assertIn("hieleria veracruz", answer)
        self.assertIn("servicios y minas de mexico actus", answer)
        self.assertNotIn("no hay registros en zoho crm ni pdfs locales sobre hieleria veracruz", answer)

    def test_todo_lo_que_debo_saber_de_hieleria_uses_real_crm_evidence(self) -> None:
        response = self.service.answer_question("dame todo lo que debo saber de Hieleria Veracruz", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("hieleria veracruz", answer)
        self.assertTrue("federico" in answer or "demo" in answer or "visita" in answer)
        self.assertNotIn("no contamos con datos previos", answer)

    def test_compound_contacts_and_tomorrow_plan_keeps_tum_scope(self) -> None:
        response = self.service.answer_question("Dame contactos y dime que hacer con tum mañana", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("tum", answer)
        self.assertNotIn("no hay registros en crm", answer)
        self.assertNotIn("no puedo entregarte datos de contacto", answer)

    def test_pending_commitments_that_i_completed_scope_to_active_seller(self) -> None:
        response = self.service.answer_question("dime que compromisos detectas pendiente o por confirmar que si realice?", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("jesus emmanuel meza", answer)
        self.assertNotIn("soporte flotimatics", answer)
        self.assertNotIn("eduardo valdez", answer)

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

    def test_specific_day_review_query_should_be_detected(self) -> None:
        intent = classify_question("que paso el 19 de marzo de 2026")
        self.assertTrue(intent.asks_for_time_review)

    def test_sales_email_draft_query_should_be_detected(self) -> None:
        intent = classify_question(
            "dame un correo electronico para mandarle a movimex, usa todo lo que sabes para tratar de lograr una venta"
        )
        self.assertTrue(intent.asks_for_sales_draft)

    def test_explicit_web_query_should_be_detected(self) -> None:
        intent = classify_question("segun la web y fuentes externas, que sabes de geotab hoy")
        self.assertTrue(intent.wants_web)

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

    def test_embedded_text_analysis_returns_summary_risks_and_next_step(self) -> None:
        response = self.service.answer_question(
            "Analiza esta nota y dime resumen, riesgos y siguiente paso:\n"
            "Se realizo visita presencial con Federico de Hieleria Veracruz. Comenta que tienen interes en una demo, "
            "pero hoy siguen usando otro sistema y no estan convencidos de cambiar todavia. "
            "Quedamos en coordinar la demo por whatsapp la proxima semana."
        )
        answer = response.answer.lower()
        self.assertIn("en corto:", answer)
        self.assertIn("riesgos detectados", answer)
        self.assertIn("siguiente paso recomendado", answer)
        self.assertIn("demo", answer)

    def test_embedded_text_with_reply_request_returns_ready_piece(self) -> None:
        response = self.service.answer_question(
            "Lee este correo y dime que respondo:\n"
            "Hola Emmanuel, gracias por la informacion. Ahorita seguimos con el proveedor actual y no tenemos urgencia, "
            "pero me interesa revisar una demo en abril si me mandas opciones. Quedo atento."
        )
        answer = response.answer.lower()
        self.assertIn("pieza lista:", answer)
        self.assertIn("siguiente paso recomendado", answer)
        self.assertIn("demo", answer)

    def test_embedded_text_question_bypasses_split_and_keeps_context(self) -> None:
        response = self.service.answer_question(
            "Analiza este texto y dime que paso, que riesgo ves y que siguiente paso recomiendas:\n"
            "Se envio propuesta el lunes 18 de marzo de 2026. El cliente comenta que ya tiene proveedor actual, "
            "pero que si le interesa comparar una prueba pequena. Pide que le marquemos el jueves para confirmar."
        )
        answer = response.answer.lower()
        self.assertIn("en corto:", answer)
        self.assertIn("riesgos detectados", answer)
        self.assertIn("fecha:", answer)
        self.assertIn("siguiente paso recomendado", answer)

    def test_owner_comparison_returns_metrics_for_both_sellers(self) -> None:
        response = self.service.answer_question("diferencias entre emmanuel y pablo melin")
        answer = response.answer.lower()
        self.assertIn("jesus emmanuel meza", answer)
        self.assertIn("pablo melin dorador", answer)
        self.assertIn("interacciones", answer)

    def test_owner_brief_lists_hot_and_cold_accounts(self) -> None:
        response = self.service.answer_question("dame clientes calientes y clientes frios", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("en corto:", answer)
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

    def test_compound_question_keeps_time_context_for_followups(self) -> None:
        parts = self.service._contextualize_subquestions(
            "que paso con movimex el 11 de marzo de 2026 y luego que comentarios hubo y que compromisos hubo",
            ["que paso con movimex el 11 de marzo de 2026", "que comentarios hubo", "que compromisos hubo"],
        )
        self.assertIn("2026-03-11", parts[1])
        self.assertIn("2026-03-11", parts[2])

    def test_compound_question_keeps_document_scope_for_followups(self) -> None:
        parts = self.service._contextualize_subquestions(
            "segun los pdfs, dame argumentos de venta para geotab y luego objeciones probables y como responderlas",
            ["dame argumentos de venta para geotab", "objeciones probables y como responderlas"],
        )
        self.assertIn("documentos internos", parts[1].lower())

    def test_fallback_split_handles_ceo_style_commas_and_followups(self) -> None:
        parts = self.service._fallback_semantic_like_split(
            "segun zoho y los pdfs, como vamos con movimex, que riesgo ves y que siguiente paso recomendarias esta semana"
        )
        self.assertGreaterEqual(len(parts), 3)
        self.assertIn("movimex", parts[0].lower())
        self.assertIn("riesgo", parts[1].lower())
        self.assertIn("siguiente paso", parts[2].lower())

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

    def test_executive_entity_brief_uses_decision_style(self) -> None:
        response = self.service.answer_question("dame un resumen ejecutivo de movimex")
        answer = response.answer.lower()
        self.assertIn("resumen ejecutivo de movimex", answer)
        self.assertIn("decision sugerida", answer)
        self.assertIn("siguiente paso recomendado", answer)

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

    def test_document_search_returns_pdf_chunks_for_geotab_product_question(self) -> None:
        with self.service._managed_connection() as conn:
            chunks = self.service._document_search(conn, "que dice geotab sobre seguridad y mantenimiento")
        self.assertTrue(chunks)
        flattened = " ".join(f"{row.get('file_name', '')} {row.get('content', '')}" for row in chunks).lower()
        self.assertIn("geotab", flattened)

    def test_document_focused_question_returns_internal_sources_in_answer(self) -> None:
        response = self.service.answer_question("segun los pdfs, que beneficios de flotimatics ayudan a una flotilla con mas control operativo")
        answer = response.answer.lower()
        self.assertIn("fuentes internas consultadas", answer)

    def test_document_backed_sales_material_mentions_internal_support(self) -> None:
        response = self.service.answer_question("con base en los documentos internos, dame argumentos de venta para una demo de geotab")
        answer = response.answer.lower()
        self.assertIn("fuentes internas consultadas", answer)
        self.assertIn("geotab", answer)

    def test_explicit_web_question_marks_response_and_appends_sources(self) -> None:
        service = SalesAssistantService(db_path=self.service.db_path)

        class FakeResponsesAPI:
            def create(self, **kwargs):
                annotation = SimpleNamespace(
                    type="url_citation",
                    title="Geotab official newsroom",
                    url="https://example.com/geotab-news",
                )
                content = SimpleNamespace(annotations=[annotation])
                message = SimpleNamespace(type="message", content=[content])
                return SimpleNamespace(
                    output_text="En la web hay referencias recientes a seguridad y eficiencia operativa en Geotab.",
                    output=[message],
                    sources=[SimpleNamespace(title="Geotab official newsroom", url="https://example.com/geotab-news")],
                )

        class FakeChatCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="En corto: internamente hay soporte documental y, como complemento, la web aporta contexto externo reciente."))]
                )

        service.client = SimpleNamespace(
            responses=FakeResponsesAPI(),
            chat=SimpleNamespace(completions=FakeChatCompletions()),
        )

        response = service.answer_question("segun la web y los documentos internos, que dice geotab sobre seguridad")
        answer = response.answer.lower()
        self.assertTrue(response.used_web)
        self.assertIn("fuentes web consultadas", answer)
        self.assertIn("https://example.com/geotab-news", answer)
        self.assertTrue(any(source.startswith("web:") for source in response.sources))

    def test_time_window_parser_supports_specific_day(self) -> None:
        window = self.service._parse_time_window("que paso el 19 de marzo de 2026")
        self.assertIsNotNone(window)
        self.assertEqual(window.start_date.isoformat(), "2026-03-19")
        self.assertEqual(window.end_date.isoformat(), "2026-03-19")

    def test_time_window_parser_supports_weekly_windows(self) -> None:
        window = self.service._parse_time_window("que hubo esta semana")
        self.assertIsNotNone(window)
        self.assertEqual(window.granularity, "week")
        self.assertLessEqual(window.start_date, window.end_date)

    def test_time_window_parser_supports_previous_month(self) -> None:
        window = self.service._parse_time_window("que paso el mes pasado")
        self.assertIsNotNone(window)
        self.assertEqual(window.granularity, "month")
        self.assertLessEqual(window.start_date, window.end_date)

    def test_time_window_parser_supports_explicit_range(self) -> None:
        window = self.service._parse_time_window("que hubo del 18 al 19 de marzo de 2026")
        self.assertIsNotNone(window)
        self.assertEqual(window.start_date.isoformat(), "2026-03-18")
        self.assertEqual(window.end_date.isoformat(), "2026-03-19")

    def test_specific_day_review_returns_temporal_summary(self) -> None:
        response = self.service.answer_question("que paso el 19 de marzo de 2026")
        answer = response.answer.lower()
        self.assertIn("2026-03-19", answer)
        self.assertIn("movimientos detectados", answer)

    def test_specific_day_who_added_something_lists_owners(self) -> None:
        response = self.service.answer_question("quien agrego algo el 19 de marzo de 2026")
        answer = response.answer.lower()
        self.assertIn("personas que registraron movimiento", answer)
        self.assertTrue(
            "eduardo valdez" in answer
            or "pablo melin dorador" in answer
            or "ayuda consultoria" in answer
        )

    def test_specific_day_comments_return_note_snippets(self) -> None:
        response = self.service.answer_question("que comentarios hubo el 18 de marzo de 2026", user=self.emeza)
        answer = response.answer.lower()
        self.assertIn("comentarios y notas visibles", answer)
        self.assertTrue("hospital san jose" in answer or "hieleria veracruz" in answer or "autolineas de pasaje alp" in answer)

    def test_specific_day_commitments_return_tasks_or_events(self) -> None:
        response = self.service.answer_question("que compromisos hubo el 13 de marzo de 2023")
        answer = response.answer.lower()
        self.assertIn("compromisos, tareas y eventos detectados", answer)
        self.assertIn("seguimiento a correo", answer)

    def test_entity_time_review_keeps_entity_scope(self) -> None:
        response = self.service.answer_question("que paso con movimex el 11 de marzo de 2026")
        answer = response.answer.lower()
        self.assertIn("movimex", answer)
        self.assertIn("2026-03-11", answer)

    def test_last_topic_mention_for_entity_returns_demo_reference(self) -> None:
        response = self.service.answer_question("ultima vez que hablo de demo en Hieleria Veracruz")
        answer = response.answer.lower()
        self.assertIn("demo", answer)
        self.assertIn("hieleria veracruz", answer)

    def test_entity_time_gap_summary_uses_first_note_and_latest_touch(self) -> None:
        response = self.service.answer_question("cuanto tiempo paso entre la primera nota y la ultima llamada de movimex")
        answer = response.answer.lower()
        self.assertIn("movimex", answer)
        self.assertTrue("pasaron" in answer or "no pude calcular" in answer)

    def test_entity_period_change_between_months_returns_delta(self) -> None:
        response = self.service.answer_question("que cambio entre febrero y marzo en movimex")
        answer = response.answer.lower()
        self.assertIn("movimex", answer)
        self.assertIn("febrero", answer)
        self.assertIn("marzo", answer)

    def test_rule_based_combine_subresponses_creates_executive_sections(self) -> None:
        combined = self.service._rule_based_combine_subresponses(
            "dame resumen de movimex y luego riesgos y luego redactame un correo",
            [
                "dame resumen de movimex",
                "que riesgos ves en movimex",
                "redactame un correo para movimex",
            ],
            [
                self.service.answer_question("dame resumen de movimex"),
                self.service.answer_question("que riesgos ves en movimex"),
                self.service.answer_question("redactame un correo para movimex"),
            ],
        )
        self.assertIsNotNone(combined)
        lowered = combined.lower()
        self.assertIn("panorama:", lowered)
        self.assertIn("riesgos y bloqueadores:", lowered)
        self.assertIn("pieza lista:", lowered)

    def test_ceo_style_multi_topic_question_returns_integrated_sections(self) -> None:
        response = self.service.answer_question(
            "segun zoho y los pdfs, como vamos con movimex, que riesgo ves y que siguiente paso recomendarias esta semana"
        )
        answer = response.answer.lower()
        self.assertIn("movimex", answer)
        self.assertTrue("panorama:" in answer or "resumen ejecutivo de movimex" in answer)
        self.assertTrue("riesgos y bloqueadores:" in answer or "riesgos detectados" in answer)
        self.assertTrue("recomendacion:" in answer or "siguiente paso recomendado:" in answer)

    def test_ceo_style_team_question_splits_kpis_and_priority(self) -> None:
        response = self.service.answer_question(
            "dame un resumen ejecutivo del equipo comercial, kpi global de la semana de todos los vendedores y a quien deberiamos empujar primero",
            user=get_user("evaldez"),
        )
        answer = response.answer.lower()
        self.assertIn("equipo", answer)
        self.assertTrue("panorama:" in answer or "resumen ejecutivo del equipo" in answer)
        self.assertTrue("indicadores:" in answer or "kpis globales" in answer or "interacciones:" in answer)
        self.assertTrue("recomendacion:" in answer or "decision sugerida" in answer)

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

    def test_structured_owner_comparison_prefers_direct_answer_path(self) -> None:
        intent = classify_question("diferencias entre emmanuel y pablo melin")
        self.assertTrue(
            self.service._should_prefer_direct_answer(
                intent,
                {"owner_comparison": [{"owner_name": "Jesus Emmanuel Meza Guzmán"}]},
                self.service._interpret_task("diferencias entre emmanuel y pablo melin", intent, {}),
            )
        )

    def test_feedback_override_uses_high_similarity_correction(self) -> None:
        override = self.service._feedback_override(
            {
                "feedback_memory": [
                    {
                        "question": "dame correos de movimex",
                        "correction": "Debio incluir a Dena Salinas y gps@movimex.mx.",
                        "similarity": 0.97,
                    }
                ]
            }
        )
        self.assertIn("gps@movimex.mx", override)


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

    def test_feedback_can_be_saved_and_loaded(self) -> None:
        original_db = history_module.WAREHOUSE_DB
        temp_dir = Path("data/test-feedback")
        temp_dir.mkdir(parents=True, exist_ok=True)
        history_db = temp_dir / "feedback.db"
        if history_db.exists():
            history_db.unlink()
        try:
            history_module.WAREHOUSE_DB = history_db
            ensure_history_schema()
            history_id = save_history("q1", "a1", "data", ["warehouse.db"], False, username="emeza")
            feedback_id = save_feedback(history_id, "bad", username="emeza", correction="Debio responder con el contacto correcto.")
            items = fetch_feedback(username="emeza")

            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["id"], feedback_id)
            self.assertEqual(items[0]["history_id"], history_id)
            self.assertEqual(items[0]["rating"], "bad")
            self.assertIn("contacto correcto", items[0]["correction"])
        finally:
            history_module.WAREHOUSE_DB = original_db
            if history_db.exists():
                history_db.unlink()

    def test_feedback_memory_returns_similar_corrected_question(self) -> None:
        original_db = history_module.WAREHOUSE_DB
        temp_dir = Path("data/test-feedback-memory")
        temp_dir.mkdir(parents=True, exist_ok=True)
        history_db = temp_dir / "feedback_memory.db"
        if history_db.exists():
            history_db.unlink()
        try:
            history_module.WAREHOUSE_DB = history_db
            ensure_history_schema()
            history_id = save_history(
                "dame correos de movimex",
                "No encontre correos.",
                "data",
                ["warehouse.db"],
                False,
                username="emeza",
            )
            save_feedback(
                history_id,
                "bad",
                username="emeza",
                correction="Debio incluir a Dena Salinas y gps@movimex.mx.",
            )
            memories = fetch_feedback_memory("dame solo los correos y nombres de movimex", username="emeza")

            self.assertEqual(len(memories), 1)
            self.assertIn("movimex", memories[0]["question"].lower())
            self.assertIn("dena salinas", memories[0]["correction"].lower())
        finally:
            history_module.WAREHOUSE_DB = original_db
            if history_db.exists():
                history_db.unlink()


if __name__ == "__main__":
    unittest.main()
