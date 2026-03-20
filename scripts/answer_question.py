import argparse
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from assistant_core.service import SalesAssistantService


def main() -> None:
    parser = argparse.ArgumentParser(description="Responder preguntas con el asistente comercial.")
    parser.add_argument("question", help="Pregunta del usuario")
    args = parser.parse_args()

    service = SalesAssistantService()
    response = service.answer_question(args.question)

    print("MODO:", response.mode)
    print("FUENTES:", ", ".join(response.sources))
    print("WEB:", "si" if response.used_web else "no")
    print()
    print(response.answer)


if __name__ == "__main__":
    main()
