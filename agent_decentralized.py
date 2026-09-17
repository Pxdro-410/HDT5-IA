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

# DEFINICIÓN DE AGENTES ESPECIALISTAS AUTÓNOMOS

faq_agent = Agent(
    name="FAQ Agent",
    instructions=(
        "Eres el Agente Especialista en Base de Conocimientos de Parachute S.A.\n"
        "Responde preguntas sobre precios, ubicación, pagos, vestimenta o requisitos usando 'search_faqs_by_text'.\n"
        "Básate exclusivamente en los datos oficiales devueltos por la herramienta.\n"
        "Si el usuario decide agendar, reservar o pregunta por el clima: haz handoff al Weather Agent.\n"
        + STRICT_RULES
    ),
    tools=[search_faqs_by_text],
)

weather_agent = Agent(
    name="Weather Agent",
    instructions=(
        "Eres el Oficial Meteorológico y de Seguridad de Vuelo de Parachute S.A.\n"
        "Verifica las condiciones de la Drop Zone con 'check_weather_for_date' (máx 16 días).\n"
        "Si las condiciones son PROHIBIDAS (viento > 28 km/h, ráfagas > 35 km/h, lluvia > 0 mm o nubes > 75%), "
        "informa claramente que NO se puede saltar ese día y se rechaza la cita indicando los motivos técnicos.\n"
        "Si el clima es IDEAL o MARGINAL (solo tándem) y el usuario desea continuar a calendarizar: haz handoff al Booking Agent.\n"
        "Si tiene dudas de la empresa o precios: haz handoff al FAQ Agent.\n"
        + STRICT_RULES
    ),
    tools=[check_weather_for_date],
)

booking_agent = Agent(
    name="Booking Agent",
    instructions=(
        "Eres el Agente de Reservas de Parachute S.A.\n"
        "Tu función es confirmar conversacionalmente la cita de salto al cliente (solicitando su nombre, fecha acordada y hora), "
        "siempre que el Weather Agent haya validado previamente el clima como IDEAL o MARGINAL (tándem).\n"
        "Si el usuario desea consultar o cambiar la fecha de vuelo: haz handoff al Weather Agent.\n"
        "Si tiene dudas de pagos o requerimientos: haz handoff al FAQ Agent.\n"
        + STRICT_RULES
    ),
)

triage_agent = Agent(
    name="Triage Agent",
    instructions=(
        "Eres el Agente de Recepción y Bienvenida de Parachute S.A.\n"
        "Tu función es identificar la intención del usuario y transferir inmediatamente mediante HANDOFF:\n"
        "- Si tiene dudas generales, precios o políticas: haz handoff al FAQ Agent.\n"
        "- Si desea saltar, ver el clima o calendarizar una cita: haz handoff al Weather Agent.\n"
        + STRICT_RULES
    ),
    handoffs=[faq_agent, weather_agent],
)

# Configurar transferencias cruzadas entre pares (Peer-to-Peer)
faq_agent.set_handoffs([weather_agent, triage_agent])
weather_agent.set_handoffs([booking_agent, faq_agent])
booking_agent.set_handoffs([weather_agent, faq_agent])


def main():
    client = get_llm_client()
    model = get_model_name()
    active_agent = triage_agent
    messages = []

    print(" ")
    print("  PARACHUTE S.A. - ARQUITECTURA DESCENTRALIZADA (handoffs)")
    print("  Coreografía Peer-to-Peer entre agentes autónomos")
    print("  Modelo:", model)
    print(" ")
    print("Escribe tu solicitud. Para salir escribe 'bye'.\n")

    while True:
        try:
            user_input = input("Usuario: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("bye", "salir", "exit"):
                print(f"\n[{active_agent.name}]: Hasta luego. Gracias por comunicarse con Parachute S.A.")
                break

            messages.append({"role": "user", "content": user_input})

            # Bucle de interacción del agente activo con soporte para handoffs
            while True:
                result = active_agent.run_turn(messages, client, model)
                if result["handoff_to"]:
                    active_agent = result["handoff_to"]
                    continue  # El nuevo agente activo toma el turno de respuesta inmediatamente
                else:
                    print(f"\n[{active_agent.name}]: {result['content']}\n")
                    break

        except KeyboardInterrupt:
            print(f"\n\n[{active_agent.name}]: Sesión finalizada.")
            break


if __name__ == "__main__":
    main()
