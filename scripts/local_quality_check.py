import argparse
import json
import sqlite3
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from assistant_core.auth import get_user
from assistant_core.service import SalesAssistantService


def fetch_samples(db_path: Path) -> dict[str, list[str]]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    owners = [
        row["owner_name"]
        for row in conn.execute(
            """
            SELECT owner_name
            FROM interactions
            WHERE owner_name IS NOT NULL
            GROUP BY owner_name
            ORDER BY COUNT(*) DESC
            LIMIT 8
            """
        ).fetchall()
    ]
    entities = [
        row["name"]
        for row in conn.execute(
            """
            SELECT related_name AS name
            FROM interactions
            WHERE related_name IS NOT NULL AND trim(related_name) <> ''
            GROUP BY related_name
            ORDER BY MAX(interaction_at) DESC
            LIMIT 16
            """
        ).fetchall()
    ]
    note_entities = [
        row["name"]
        for row in conn.execute(
            """
            SELECT parent_name AS name
            FROM notes
            WHERE parent_name IS NOT NULL AND trim(parent_name) <> ''
            GROUP BY parent_name
            ORDER BY MAX(created_time) DESC
            LIMIT 12
            """
        ).fetchall()
    ]
    conn.close()

    merged_entities: list[str] = []
    for value in entities + note_entities:
        if value and value not in merged_entities:
            merged_entities.append(value)
    return {
        "owners": owners[:6],
        "entities": merged_entities[:12],
    }


def build_questions(samples: dict[str, list[str]]) -> list[dict[str, str]]:
    entities = samples["entities"]
    owners = samples["owners"]
    questions: list[dict[str, str]] = []

    base_self_questions = [
        {"username": "emeza", "question": "dame todo lo que debo saber de mis contactos o leads"},
        {"username": "emeza", "question": "kpi mio de la semana"},
        {"username": "emeza", "question": "dame clientes calientes y clientes frios"},
        {"username": "emeza", "question": "a quien debo llamar hoy y por que"},
        {"username": "emeza", "question": "a quien debo contactar hoy"},
        {"username": "emeza", "question": "en base a mis notas a quien me recomiendas llamar hoy"},
        {"username": "emeza", "question": "ultima nota agregada"},
        {"username": "emeza", "question": "compromisos pendientes para hoy"},
        {"username": "emeza", "question": "a quien le hable ayer"},
        {"username": "emeza", "question": "mis kpis"},
        {"username": "emeza", "question": "analiza mis notas y arma un plan para hoy"},
        {"username": "emeza", "question": "en base a mis notas, que me recomiendas hacer hoy"},
        {"username": "emeza", "question": "que oportunidades tengo mas calientes"},
        {"username": "emeza", "question": "donde estoy dejando dinero en la mesa"},
        {"username": "emeza", "question": "dame mis oportunidades mas fuertes y escribe un correo para la principal"},
        {"username": "emeza", "question": "dame mis clientes calientes y luego armame mensaje para el mejor"},
    ]
    questions.extend(base_self_questions)

    for owner in owners:
        questions.extend(
            [
                {"username": "evaldez", "question": f"kpi {owner} de la semana"},
                {"username": "evaldez", "question": f"quienes son los clientes de {owner}"},
                {"username": "evaldez", "question": f"actividad reciente de {owner}"},
                {"username": "evaldez", "question": f"diferencias entre {owner} y Eduardo Valdez"},
            ]
        )

    for entity in entities:
        questions.extend(
            [
                {"username": "emeza", "question": entity},
                {"username": "emeza", "question": f"dame todo lo que debo saber de {entity}"},
                {"username": "emeza", "question": f"kpi {entity} de la semana"},
                {"username": "emeza", "question": f"cuales son los nombres de contactos, numeros de telefono y correos de {entity}"},
                {"username": "emeza", "question": f"que objeciones hay en {entity}"},
                {"username": "emeza", "question": f"que sigue con {entity}"},
                {"username": "emeza", "question": f"ultimo contacto de {entity}"},
                {"username": "emeza", "question": f"redactame un correo para {entity} buscando cerrar una demo"},
                {"username": "emeza", "question": f"hazme argumentos de venta para una llamada con {entity}"},
                {"username": "emeza", "question": f"que harias hoy, manana y esta semana con {entity}"},
                {"username": "emeza", "question": f"dame resumen de {entity} y luego redactame un correo"},
            ]
        )

    for first, second in zip(entities[::2], entities[1::2]):
        questions.extend(
            [
                {"username": "emeza", "question": f"compara {first} vs {second}"},
                {"username": "emeza", "question": f"cual va mas avanzado entre {first} y {second}"},
            ]
        )

    while len(questions) < 110:
        entity = entities[len(questions) % len(entities)]
        questions.append(
            {
                "username": "emeza",
                "question": f"hazme un resumen ejecutivo de {entity}",
            }
        )
    return questions[:110]


def evaluate_questions(db_path: Path, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = fetch_samples(db_path)
    questions = build_questions(samples)
    service = SalesAssistantService(db_path=str(db_path))

    results = []
    for item in questions:
        user = get_user(item["username"])
        response = service.answer_question(item["question"], user=user)
        answer_lower = response.answer.lower()
        issues = []
        if "no encontre" in answer_lower or "no hay evidencia" in answer_lower:
            issues.append("fallback_negative")
        if len(response.answer.strip()) < 60:
            issues.append("too_short")
        results.append(
            {
                "username": item["username"],
                "question": item["question"],
                "mode": response.mode,
                "sources": response.sources,
                "answer": response.answer,
                "issues": issues,
            }
        )

    report_path = output_dir / "local_quality_report.md"
    json_path = output_dir / "local_quality_results.json"
    json_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    flagged = [item for item in results if item["issues"]]
    lines = [
        "# Local Quality Report",
        "",
        f"- Preguntas evaluadas: {len(results)}",
        f"- Respuestas con bandera heurística: {len(flagged)}",
        "",
        "## Ejemplos destacados",
        "",
    ]
    for item in results[:20]:
        lines.extend(
            [
                f"### {item['question']}",
                f"- Usuario: {item['username']}",
                f"- Modo: {item['mode']}",
                f"- Issues: {', '.join(item['issues']) if item['issues'] else 'ninguno'}",
                "",
                item["answer"],
                "",
            ]
        )
    if flagged:
        lines.extend(["## Respuestas a revisar", ""])
        for item in flagged[:30]:
            lines.append(f"- {item['question']} -> {', '.join(item['issues'])}")
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local assistant quality check with 100+ prompts.")
    parser.add_argument("--db-path", required=True, help="Path to warehouse.db")
    parser.add_argument("--output-dir", default="data/local-eval", help="Directory for generated reports")
    args = parser.parse_args()

    report_path = evaluate_questions(Path(args.db_path), Path(args.output_dir))
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
