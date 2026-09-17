import json
import urllib.request
from datetime import datetime, date, timedelta
from typing import Any, Dict

DROP_ZONE_LAT = 14.013722
DROP_ZONE_LON = -90.771611
MAX_FORECAST_DAYS = 16


def check_weather_for_date(fecha: str) -> Dict[str, Any]:
    """
    Consulta la API gratuita de Open-Meteo (sin necesidad de API Key) para las coordenadas
    de aterrizaje de Parachute S.A. (14.013722, -90.771611) y evalúa los criterios de seguridad.
    """
    today = date.today()
    cleaned = fecha.strip().lower()

    if cleaned in ("hoy", "today"):
        target_date = today
    elif cleaned in ("mañana", "manana", "tomorrow"):
        target_date = today + timedelta(days=1)
    else:
        try:
            target_date = datetime.strptime(cleaned, "%Y-%m-%d").date()
        except ValueError:
            return {
                "valido": False,
                "error": "formato_invalido",
                "mensaje": f"Formato de fecha inválido '{fecha}'. Por favor utiliza el formato AAAA-MM-DD (ejemplo: {today.isoformat()})."
            }

    diff_days = (target_date - today).days

    if diff_days < 0:
        return {
            "valido": False,
            "error": "fecha_pasada",
            "mensaje": f"La fecha solicitada ({target_date.isoformat()}) ya pasó. Selecciona una fecha presente o futura."
        }

    # Restricción de 16 días máxima de Open-Meteo especificada en la guía
    if diff_days > MAX_FORECAST_DAYS:
        return {
            "valido": False,
            "error": "excede_16_dias",
            "max_dias": MAX_FORECAST_DAYS,
            "mensaje": (
                f"No es posible consultar el clima para el {target_date.isoformat()} porque la API de Open-Meteo "
                f"únicamente provee hasta {MAX_FORECAST_DAYS} días de predicción. "
                f"Por favor selecciona una fecha entre hoy ({today.isoformat()}) y el {(today + timedelta(days=MAX_FORECAST_DAYS)).isoformat()}."
            )
        }

    url = (
        f"https://api.open-meteo.com/v1/forecast?latitude={DROP_ZONE_LAT}&longitude={DROP_ZONE_LON}"
        f"&daily=temperature_2m_max,wind_speed_10m_max,wind_gusts_10m_max,precipitation_sum,cloud_cover_mean"
        f"&timezone=auto&forecast_days=16"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "ParachuteSA-Agent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        return {
            "valido": False,
            "error": "api_error",
            "mensaje": f"Error al conectar con el servicio público de Open-Meteo: {e}"
        }

    daily = data.get("daily", {})
    dates = daily.get("time", [])
    target_str = target_date.isoformat()

    if target_str not in dates:
        return {
            "valido": False,
            "error": "sin_datos",
            "mensaje": f"No se encontraron predicciones meteorológicas para {target_str}."
        }

    idx = dates.index(target_str)
    wind_speed = float(daily["wind_speed_10m_max"][idx] or 0.0)
    wind_gusts = float(daily["wind_gusts_10m_max"][idx] or 0.0)
    precipitation = float(daily["precipitation_sum"][idx] or 0.0)
    cloud_cover = float(daily["cloud_cover_mean"][idx] or 0.0)
    temperature = float(daily["temperature_2m_max"][idx] or 25.0)

    # Evaluación de criterios de seguridad de Parachute S.A.
    reasons_prohibited = []
    reasons_marginal = []

    # 1. Viento superficie
    if wind_speed > 28.0:
        reasons_prohibited.append(f"Viento en superficie de {wind_speed:.1f} km/h supera los 28 km/h (muy difícil de controlar el salto).")
    elif 20.0 <= wind_speed <= 28.0:
        reasons_marginal.append(f"Viento en superficie de {wind_speed:.1f} km/h (20-28 km/h: permitido SOLO tándem con instructor experimentado).")

    # 2. Ráfagas de viento
    if wind_gusts > 35.0:
        reasons_prohibited.append(f"Ráfagas de viento de {wind_gusts:.1f} km/h superan el límite de 35 km/h.")

    # 3. Precipitación
    if precipitation > 0.0:
        reasons_prohibited.append(f"Precipitación de {precipitation:.1f} mm detectada (lluvia prohibida: daña el equipo y lastima la piel en caída libre).")

    # 4. Visibilidad / Cobertura de nubes
    if cloud_cover > 75.0:
        reasons_prohibited.append(f"Cobertura de nubes del {cloud_cover:.0f}% supera el 75% (techo bajo impide reglas de vuelo visual VFR).")
    elif 30.0 <= cloud_cover <= 75.0:
        reasons_marginal.append(f"Cobertura de nubes del {cloud_cover:.0f}% (30-75%: nubes dispersas).")

    if reasons_prohibited:
        status = "NO SEGURO / PROHIBIDO"
        apt = False
    elif reasons_marginal:
        status = "MARGINAL (Solo tándem experimentado)"
        apt = True
    else:
        status = "IDEAL (Condiciones óptimas)"
        apt = True

    return {
        "valido": True,
        "fecha": target_str,
        "estado": status,
        "apto_para_salto": apt,
        "viento_superficie_kmh": wind_speed,
        "rafagas_kmh": wind_gusts,
        "lluvia_mm": precipitation,
        "cobertura_nubes_pct": cloud_cover,
        "temperatura_max_c": temperature,
        "motivos_prohibicion": reasons_prohibited,
        "advertencias_marginales": reasons_marginal
    }
