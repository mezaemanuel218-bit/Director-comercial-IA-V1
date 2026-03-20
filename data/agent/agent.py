import json
import os
from openai import OpenAI
from dotenv import load_dotenv

from agent.tools import (
    analyze_client,
    kpi_contacts_last_week,
    owners_stats
)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_client",
            "description": "Analiza un cliente usando notas CRM",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"}
                },
                "required": ["name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "kpi_contacts_last_week",
            "description": "Contactos agregados última semana",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "owners_stats",
            "description": "Cuantos clientes tiene cada vendedor",
            "parameters": {
                "type": "object",
                "properties": {}
            }
        }
    }
]


def ask(question):

    messages = [
        {
            "role": "system",
            "content": """
Eres un director comercial experto.

Puedes:

analizar clientes
consultar KPIs
revisar actividad CRM
dar estrategia comercial
"""
        },
        {
            "role": "user",
            "content": question
        }
    ]

    r = client.chat.completions.create(
        model="gpt-4.1",
        messages=messages,
        tools=TOOLS
    )

    msg = r.choices[0].message

    if msg.tool_calls:

        tool = msg.tool_calls[0]

        name = tool.function.name

        args = json.loads(tool.function.arguments)

        if name == "analyze_client":
            result = analyze_client(**args)

        elif name == "kpi_contacts_last_week":
            result = kpi_contacts_last_week()

        elif name == "owners_stats":
            result = owners_stats()

        else:
            result = "tool desconocida"

        messages.append(msg)

        messages.append({
            "role": "tool",
            "tool_call_id": tool.id,
            "content": str(result)
        })

        r2 = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages
        )

        return r2.choices[0].message.content

    return msg.content