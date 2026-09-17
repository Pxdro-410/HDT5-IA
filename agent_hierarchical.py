from services import (
    Agent,
    check_weather_for_date,
    search_faqs_by_text,
    get_llm_client,
    get_model_name,
)

# reglas de comportamiento para el asistente

STRICT_RULES = """
REGLAS OBLIGATORIAS:
- ESTRICTAMENTE PROHIBIDO EL USO DE EMOJIS. No utilices ningún emoji o símbolo decorativo bajo ninguna circunstancia.
- Tu única fuente de información son las herramientas. No agregues conocimiento externo, no inventes justificaciones biológicas/médicas ni condiciones que no aparezcan en los resultados devueltos por las herramientas.
- Si una herramienta devuelve datos válidos para una fecha, NO digas que la fecha está fuera del horizonte de pronóstico; básate estrictamente en el campo 'valido' del resultado.
- Responde de forma formal, sobria, directa y concisa.
"""

# NIVEL 3: AGENTES EJECUTORES / WORKERS

faq_worker = Agent(
    name="FAQ Worker",
    instructions=(
        "Consulta la base de conocimientos con 'search_faqs_by_text' y reporta las respuestas oficiales.\n"
        "No agregues explicaciones externas ni modifiques el contenido recuperado.\n"
        + STRICT_RULES
    ),
    tools=[search_faqs_by_text],
)

weather_worker = Agent(
    name="Weather Worker",
    instructions=(
        "Consulta Open-Meteo con 'check_weather_for_date' y reporta el dictamen de seguridad: "
        "IDEAL, MARGINAL o NO SEGURO / PROHIBIDO.\n"
        "Reporta estrictamente los datos entregados por la herramienta.\n"
        + STRICT_RULES
    ),
    tools=[check_weather_for_date],
)

# NIVEL 2: MANAGERS INTERMEDIOS (Utilizan as_tool para sus subordinados)

cs_manager = Agent(
    name="Customer Service Manager",
    instructions=(
        "Eres el Manager de Atención al Cliente de Parachute S.A.\n"
        "Gestionas dudas sobre políticas, precios, vestimenta, peso o requisitos.\n"
        "Ordenas a tu subordinado 'faq_expert' mediante as_tool() buscar la información.\n"
        "Reporta la respuesta sin inventar datos ni agregar emojis.\n"
        + STRICT_RULES
    ),
    tools=[
        faq_worker.as_tool(
            tool_name="faq_expert",
            tool_description="Handles Parachute S.A. FAQ inquiries."
        )
    ],
)

ops_manager = Agent(
    name="Operations Manager",
    instructions=(
        "Eres el Manager de Operaciones y Reservas de Parachute S.A.\n"
        "Tu protocolo estricto para calendarizar citas de salto:\n"
        "1. Ordena al Oficial de Clima con 'weather_expert' verificar la fecha solicitada.\n"
        "2. Si el clima es 'NO SEGURO / PROHIBIDO', RECHAZA la cita y reporta las razones técnicas de veto.\n"
        "3. Si es 'MARGINAL', acepta la cita indicando que es EXCLUSIVA para modalidad TÁNDEM con instructor experimentado.\n"
        "4. Si es 'IDEAL', ACEPTA y confirma la cita de salto para esa fecha.\n"
        "Devuelve un reporte operacional consolidado al Director General.\n"
        + STRICT_RULES
    ),
    tools=[
        weather_worker.as_tool(
            tool_name="weather_expert",
            tool_description="Handles drop-zone weather audits via Open-Meteo."
        ),
    ],
)

# NIVEL 1: DIRECTOR GENERAL (Utiliza as_tool para comandar a sus Managers)

DIRECTOR_INSTRUCTIONS = f"""Eres el DIRECTOR GENERAL de Parachute S.A.
Lideras la ARQUITECTURA JERÁRQUICA multinivel.
Interactúas con el cliente y delegas tareas a tus Managers de Nivel 2 mediante as_tool():
- 'customer_service': Para dudas de la empresa, precios o FAQs.
- 'operations_booking': Para solicitudes de reserva y calendarización de salto.

Tus Managers coordinan a sus propios workers ejecutores (Nivel 3).
Tu función es recibir sus reportes consolidados y responder de forma ejecutiva, formal, concisa y sobria al cliente.

{STRICT_RULES}
"""

director = Agent(
    name="Director General",
    instructions=DIRECTOR_INSTRUCTIONS,
    tools=[
        cs_manager.as_tool(
            tool_name="customer_service",
            tool_description="Delegates questions and customer support to the Customer Service Manager."
        ),
        ops_manager.as_tool(
            tool_name="operations_booking",
            tool_description="Delegates flight audit and reservation workflows to the Operations Manager."
        ),
    ],
)


def main():
    client = get_llm_client()
    model = get_model_name()
    messages = []

    print(" ")
    print("  PARACHUTE S.A. - ARQUITECTURA JERÁRQUICA (as_tool)")
    print("  Director General (N1) -> Managers (N2) -> Workers (N3)")
    print("  Modelo:", model)
    print(" ")
    print("Escribe tu solicitud. Para salir escribe 'bye'.\n")

    while True:
        try:
            user_input = input("Usuario: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("bye", "salir", "exit"):
                print("\nDirector General: Gracias por comunicarse con Parachute S.A. Hasta luego.")
                break

            messages.append({"role": "user", "content": user_input})
            result = director.run_turn(messages, client, model)
            print(f"\nDirector General: {result['content']}\n")

        except KeyboardInterrupt:
            print("\n\nDirector General: Sesión finalizada.")
            break


if __name__ == "__main__":
    main()
