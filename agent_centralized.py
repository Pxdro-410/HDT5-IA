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

# AGENTES ESPECIALISTAS SUBORDINADOS

faq_agent = Agent(
    name="FAQ Agent",
    instructions=(
        "Eres el especialista en preguntas frecuentes de Parachute S.A.\n"
        "Tu única fuente confiable es la herramienta 'search_faqs_by_text'.\n"
        "Responde de forma textual y concisa basándote ÚNICAMENTE en la respuesta oficial recuperada.\n"
        "No agregues recomendaciones externas, no elabores teorías médicas ni agregues datos no presentes en el corpus.\n"
        + STRICT_RULES
    ),
    tools=[search_faqs_by_text],
)

weather_agent = Agent(
    name="Weather Agent",
    instructions=(
        "Eres el oficial meteorológico de Parachute S.A.\n"
        "Consulta las condiciones en la Drop Zone usando 'check_weather_for_date'.\n"
        "Reporta con exactitud los valores obtenidos y el dictamen de seguridad: "
        "IDEAL, MARGINAL (solo tándem experimentado) o NO SEGURO / PROHIBIDO "
        "(viento > 28 km/h, ráfagas > 35 km/h, lluvia > 0.0 mm o nubes > 75%).\n"
        "Si la herramienta indica que excede 16 días, informa ese error. Si la herramienta retorna datos válidos, reporta esos datos sin dudar de la fecha.\n"
        + STRICT_RULES
    ),
    tools=[check_weather_for_date],
)

# SUPERVISOR CENTRAL (Utiliza as_tool para incluir a los especialistas)

SUPERVISOR_INSTRUCTIONS = f"""Eres el SUPERVISOR CENTRAL de Parachute S.A.
En esta arquitectura CENTRALIZADA (Hub & Spoke), eres el único punto de contacto con el usuario.
Orquestas a tus agentes especialistas mediante las herramientas generadas con as_tool():
- 'faq_expert': Para resolver dudas sobre la empresa, costos, ubicaciones o requisitos.
- 'weather_expert': Para auditar el clima en la zona de aterrizaje (Open-Meteo, máx 16 días).

CALENDARIZACIÓN DE CITAS:
Cuando el usuario solicite calendarizar una cita de salto en paracaídas:
1. Consulta obligatoriamente a 'weather_expert' para la fecha solicitada.
2. Si el dictamen es 'NO SEGURO / PROHIBIDO', RECHAZA la cita informando de manera técnica y concisa los parámetros que superan los límites de seguridad.
3. Si el dictamen es 'MARGINAL', informa que la cita se acepta ÚNICAMENTE para salto en modalidad TÁNDEM con instructor experimentado.
4. Si el dictamen es 'IDEAL', ACEPTA y confirma la cita para la fecha y hora indicadas.

{STRICT_RULES}
"""

supervisor = Agent(
    name="Supervisor Central",
    instructions=SUPERVISOR_INSTRUCTIONS,
    tools=[
        faq_agent.as_tool(
            tool_name="faq_expert",
            tool_description="Handles Parachute S.A. FAQ questions and company information requests."
        ),
        weather_agent.as_tool(
            tool_name="weather_expert",
            tool_description="Handles drop-zone weather checks and flight safety evaluation via Open-Meteo."
        ),
    ],
)


def main():
    client = get_llm_client()
    model = get_model_name()
    messages = []

    print(" ")
    print("  PARACHUTE S.A. - ARQUITECTURA CENTRALIZADA (as_tool)")
    print("  Supervisor Central orquestando especialistas como herramientas")
    print("  Modelo:", model)
    print(" ")
    print("Escribe tu consulta o solicitud de cita. Para salir escribe 'bye'.\n")

    while True:
        try:
            user_input = input("Usuario: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("bye", "salir", "exit"):
                print("\nSupervisor Central: Hasta luego. Gracias por comunicarse con Parachute S.A.")
                break

            messages.append({"role": "user", "content": user_input})
            result = supervisor.run_turn(messages, client, model)
            print(f"\nSupervisor Central: {result['content']}\n")

        except KeyboardInterrupt:
            print("\n\nSupervisor Central: Sesión finalizada.")
            break


if __name__ == "__main__":
    main()
