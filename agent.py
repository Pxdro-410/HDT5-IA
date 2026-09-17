import json
import os
import sys
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from db import search_faqs

SYSTEM_PROMPT = """Eres el asistente oficial de preguntas frecuentes de la empresa Parachute S.A.
Tu objetivo es responder las dudas de los usuarios sobre el evento y los servicios de paracaidismo.

REGLAS OBLIGATORIAS:
- Tu única fuente confiable de conocimiento es la base de datos de Parachute S.A.
- Para responder cualquier pregunta sobre eventos, requisitos, fechas, ubicaciones, pagos, políticas o servicios, DEBES usar la herramienta 'consultar_base_conocimientos'.
- Responde ÚNICAMENTE basándote en la información devuelta por la herramienta.
- Si la información devuelta no responde la duda del usuario, o si la pregunta es sobre un tema ajeno o no contemplado en el corpus, debes admitir claramente que no cuentas con esa información y sugerir que se comunique directamente con Parachute S.A. (soporte@parachutesa.gt).
- No inventes información ni utilices conocimiento externo que no figure en los resultados recuperados.
- Responde de forma cordial, con tonalidad formal y se conciso en las respuestas.
"""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "consultar_base_conocimientos",
            "description": "Busca información relevante en la base de datos vectorial de preguntas frecuentes (FAQs) de Parachute S.A.",
            "parameters": {
                "type": "object",
                "properties": {
                    "consulta": {
                        "type": "string",
                        "description": "La duda, término clave o pregunta para buscar en el corpus oficial de Parachute S.A.",
                    }
                },
                "required": ["consulta"],
            },
        },
    }
]


def build_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("NVIDIA_API_KEY")
    base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    if not api_key:
        print("Error: falta NVIDIA_API_KEY en el archivo .env.")
        sys.exit(1)
    
    # Limpiar posibles comillas o espacios residuales
    api_key = api_key.strip().strip('"').strip("'")
    return OpenAI(api_key=api_key, base_url=base_url)


def execute_tool_call(tool_call, embed_model: SentenceTransformer) -> str:
    func_name = tool_call.function.name
    if func_name == "consultar_base_conocimientos":
        try:
            args = json.loads(tool_call.function.arguments)
        except Exception:
            args = {}
        query = args.get("consulta", "")
        
        # Generar embedding de la consulta del usuario
        query_vector = embed_model.encode(query, normalize_embeddings=True).tolist()
        results = search_faqs(query_vector, limit=3)

        if not results:
            return json.dumps({"resultado": "No se encontraron registros relacionados en la base de datos."})

        docs = []
        for r in results:
            docs.append({
                "faq_id": r["faq_id"],
                "categoria": r["category"],
                "pregunta": r["question"],
                "respuesta": r["answer"],
                "similitud": round(float(r["similarity"]), 4),
            })
        return json.dumps({"faqs_relevantes": docs}, ensure_ascii=False)

    return json.dumps({"error": f"Herramienta desconocida '{func_name}'"})


def main() -> None:
    client = build_client()
    model = os.getenv("NVIDIA_MODEL", "moonshotai/kimi-k3")

    print("Cargando modelo local de embeddings...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT}
    ]

    print("\n ")
    print("HDT4 - Pedro Caso")
    print("Agente de FAQs para Parachute S.A.")
    print("Conectado a PostgreSQL y pgvector con Tool Calling")
    print(" ")
    print("Escribe tu pregunta.")
    print("Para salir escribe: 'bye' o presiona Ctrl-C.\n")

    try:
        while True:
            user_input = input("Usuario: ").strip()
            if not user_input:
                continue
            if user_input.lower() == "bye":
                print("\nAgente AI: ¡Hasta luego! Gracias por consultar con Parachute S.A.")
                break

            messages.append({"role": "user", "content": user_input})

            # Primera llamada al LLM enviando las herramientas disponibles
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
            )
            response_msg = response.choices[0].message

            # Verificar si el LLM decidió invocar una función/herramienta
            if response_msg.tool_calls:
                messages.append(response_msg)

                for tool_call in response_msg.tool_calls:
                    tool_result_str = execute_tool_call(tool_call, embed_model)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "content": tool_result_str,
                    })

                # Segunda llamada para generar la respuesta final fundamentada en el resultado de la tool
                final_response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                )
                answer = final_response.choices[0].message.content
            else:
                answer = response_msg.content

            messages.append({"role": "assistant", "content": answer})
            print(f"\nAgente AI: {answer}\n")

    except KeyboardInterrupt:
        print("\n\nAgente AI: ¡Hasta luego! Gracias por consultar con Parachute S.A.")


if __name__ == "__main__":
    main()