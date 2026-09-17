# Hoja de Trabajo 5: Orquestacion de Sistemas Multiagente (MAS)

## Informacion del Proyecto
* **Curso:** AI Engineering
* **Institucion:** Universidad del Valle de Guatemala (UVG)
* **Autor:** Pedro Caso
* **Tema:** Sistemas Multiagente (MAS) bajo Arquitecturas Centralizada, Jerarquica y Descentralizada

---

## 1. Descripcion General

Parachute S.A. es una empresa especializada en experiencias de paracaidismo en Guatemala. Previamente, la empresa contaba con un sistema RAG (Retrieval-Augmented Generation) para responder preguntas frecuentes (FAQs) utilizando una base de datos vectorial en PostgreSQL con extension pgvector.

Para esta fase, la empresa solicito ampliar sus capacidades operativas mediante la incorporacion de calendarizacion de citas para saltos en paracaidas, supeditada de manera obligatoria a una auditoria meteorologica en tiempo real. Debido a la naturaleza de alto riesgo del paracaidismo, ninguna cita puede ser confirmada sin antes auditar los parametros atmosfericos de la zona de aterrizaje (Drop Zone) a traves de la API publica de Open-Meteo.

El objetivo principal de esta practica es resolver este flujo de negocio implementando y comparando tres arquitecturas de orquestacion multiagente:
1. **Arquitectura Centralizada**
2. **Arquitectura Jerarquica**
3. **Arquitectura Descentralizada**

---

## 2. Requerimientos Funcionales y Reglas de Seguridad

### Parametros de Auditoria Climatica
* **Ubicacion de la Drop Zone:** Coordenadas `14.013722, -90.771611` (Puerto San Jose, Escuintla, Guatemala).
* **Horizonte de Pronostico:** Maximo 16 dias hacia el futuro. Si el cliente solicita una fecha superior a 16 dias, el sistema debe indicar con precision que la API de Open-Meteo no provee pronosticos para dicho horizonte.
* **Fuente de Informacion:** API REST publica de Open-Meteo (sin clave de acceso requerida).

### Criterios Meteorologicos de Decision
Los parametros capturados y evaluados son los siguientes:
* **Velocidad del viento en superficie (`wind_speed_10m`):**
  * *Ideal:* Menor a 20 km/h.
  * *Marginal:* Entre 20 y 28 km/h (autorizado exclusivamente para modalidad tandem con instructor experimentado).
  * *No Seguro / Prohibido:* Mayor a 28 km/h (dificultad severa de control del salto).
* **Rafagas de viento (`wind_gust_10m`):**
  * *No Seguro / Prohibido:* Mayor a 35 km/h.
* **Precipitacion (`precipitation`):**
  * *No Seguro / Prohibido:* Mayor a 0.0 mm (la lluvia dana el tejido del paracaidas y causa lesiones por impacto en cada libre).
* **Cobertura de nubes / Visibilidad (`cloud_cover`):**
  * *Ideal:* Menor al 30% (visibilidad clara).
  * *Marginal:* Entre 30% y 75% (nubes dispersas).
  * *No Seguro / Prohibido:* Mayor al 75% (un techo bajo de nubes incumple las reglas de vuelo visual VFR y anula la visibilidad del terreno).
* **Temperatura (`temperature_2m`):** Parametro informativo para la preparacion del pasajero.

### Politica de Confirmacion de Citas
* **Dictamen IDEAL:** La cita se confirma de forma conversacional para la fecha y hora indicadas.
* **Dictamen MARGINAL:** Se informa al usuario que las condiciones son limite y que la cita se acepta unicamente bajo modalidad de salto tandem con instructor senior.
* **Dictamen NO SEGURO / PROHIBIDO:** Se rechaza categoricamente la cita y se detallan las razones tecnicas exactas que motivaron el veto.

---

## 3. Diseno y Abstraccion de Componentes (`services/`)

Para desacoplar la logica de integraciones de la logica de coordinacion multiagente, se construyo una capa modular de servicios:

```
HDT5-IA/
|-- services/
|   |-- __init__.py           # Exportacion limpia de funciones y clases
|   |-- agent_framework.py    # Framework agnostico de agentes (tools, as_tool, handoffs)
|   |-- weather_service.py    # Cliente Open-Meteo y motor de reglas de seguridad
|-- db.py                     # PostgreSQL + pgvector + all-MiniLM-L6-v2
|-- load.py                   # Script de parseo y vectorizacion de 120 FAQs
|-- agent_centralized.py      # Implementacion Arquitectura Centralizada
|-- agent_hierarchical.py     # Implementacion Arquitectura Jerarquica
|-- agent_decentralized.py    # Implementacion Arquitectura Descentralizada
|-- diagramas/                # imagenes renderizadas con Graphviz
|-- docker-compose.yml        # Contenedor pgvector
|-- requeriments.txt          # Dependencias de Python
```

### Framework Unificado de Agentes (`services/agent_framework.py`)
Implementa la clase `Agent`, la cual soporta de manera nativa:
* **Ejecucion de herramientas propias (`tools`):** Manejo recursivo de invocacion de funciones en OpenAI Function Calling.
* **Agentes como herramientas (`as_tool()`):** Empaqueta un subagente completo dentro de una especificacion de funcion, permitiendo que un supervisor orqueste a sus subordinados sin perder su contexto principal.
* **Transferencias de control (`handoffs=[...]`):** Permite a los agentes transferir el turno activo a otro par en una red descentralizada.
* **Prevencion de alucinaciones:** Prompts de sistema estrictos para garantizar apego unicamente a las fuentes devueltas por las herramientas.

---

## 4. Arquitecturas Implementadas

### A. Arquitectura Centralizada (`agent_centralized.py`)
* **Patron:** Hub and Spoke (Estrella).
* **Funcionamiento:** Un **Supervisor Central** actua como unico punto de contacto para el usuario. Dispone de dos subagentes subordinados encapsulados mediante la funcion `as_tool()`:
  * `faq_expert`: Responde consultas sobre politicas, pagos, vestimenta o requisitos mediante busqueda vectorial en PostgreSQL.
  * `weather_expert`: Audita el clima de la Drop Zone en Open-Meteo y emite el dictamen de seguridad.
* **Diagrama:** `diagramas/arquitectura_centralizada.png`

### B. Arquitectura Jerarquica (`agent_hierarchical.py`)
* **Patron:** Arbol Multinivel con Cadena de Mando.
* **Funcionamiento:** Se estructura en tres niveles jerarquicos:
  * **Nivel 1 (Direccion Estrategica):** El `Director General` atiende al usuario y mantiene la vision ejecutiva del negocio.
  * **Nivel 2 (Managers de Dominio):**
    * `Customer Service Manager`: Responsable del area de atencion al cliente y dudas de empresa.
    * `Operations Manager`: Responsable de la auditoria de vuelo y el protocolo de autorizacion de saltos.
  * **Nivel 3 (Workers Ejecutores):**
    * `FAQ Worker`: Ejecuta la herramienta de busqueda semantica `search_faqs_by_text`.
    * `Weather Worker`: Ejecuta la herramienta de auditoria `check_weather_for_date`.
* Cada nivel superior comanda a su nivel inferior mediante `as_tool()`.
* **Diagrama:** `diagramas/arquitectura_jerarquica.png`

### C. Arquitectura Descentralizada (`agent_decentralized.py`)
* **Patron:** Coreografia Peer-to-Peer mediante transferencias (`handoffs`).
* **Funcionamiento:** No existe un orquestador central permanente. Cuatro agentes especializados colaboran transfiriendose el control de la conversacion:
  * `Triage Agent`: Agente de recepcion que clasifica la intencion inicial y transfiere al agente correspondiente.
  * `FAQ Agent`: Resuelve dudas de conocimientos y, si el usuario decide saltar, realiza un `handoff` a `Weather Agent`.
  * `Weather Agent`: Evalua el clima. Si es apto (Ideal o Marginal), realiza un `handoff` a `Booking Agent`; si el usuario tiene dudas de la empresa, transfiere a `FAQ Agent`.
  * `Booking Agent`: Formaliza conversacionalmente la cita con los datos del usuario.
* **Diagrama:** `diagramas/arquitectura_descentralizada.png`

---

## 5. Instalacion y Configuracion

### Requisitos Previos
* Python 3.10 o superior instalado.
* Docker Desktop instalado y en ejecucion (para la base de datos PostgreSQL con pgvector).

### Paso 1: Clonar el Repositorio y Configurar el Entorno
```bash
git clone https://github.com/Pxdro-410/HDT5-IA.git
cd HDT5-IA

# Crear y activar entorno virtual
python -m venv .venv
# En Windows:
.venv\Scripts\activate
# En Linux/macOS:
source .venv/bin/activate

# Instalar dependencias
pip install -r requeriments.txt
```

### Paso 2: Variables de Entorno (`.env`)
Crear un archivo `.env` en la raiz del proyecto con la siguiente configuracion:
```ini
OPENAI_API_KEY=tu_api_key_de_openai
MODEL_NAME=gpt-4o-mini

DB_HOST=localhost
DB_PORT=5432
DB_NAME=parachute_faqs
DB_USER=parachute_user
DB_PASSWORD=parachute_secret
```

### Paso 3: Levantar la Base de Datos Vectorial
```bash
docker-compose up -d
```

### Paso 4: Cargar y Vectorizar las FAQs Oficiales
Ejecutar el script de carga que lee las 120 preguntas oficiales de Parachute S.A., genera los embeddings con `all-MiniLM-L6-v2` e inserta los registros en PostgreSQL:
```bash
python load.py
```

---

## 7. Ejecucion de los Programas

Cada uno de los tres programas resuelve el mismo problema de negocio de forma autonoma a traves de su respectiva arquitectura:

### Ejecucion de la Arquitectura Centralizada
```bash
python agent_centralized.py
```

### Ejecucion de la Arquitectura Jerarquica
```bash
python agent_hierarchical.py
```

### Ejecucion de la Arquitectura Descentralizada
```bash
python agent_decentralized.py
```

---

## 8. Video Demostrativo

El video explicativo con la demostracion de funcionamiento de las tres arquitecturas, validacion de consultas de FAQs, auditoria meteorologica con Open-Meteo y rechazo/confirmacion de citas se encuentra disponible en el siguiente enlace:

* **Enlace al Video Demostrativo:** https://youtu.be/Cb9t7p0z5Nw
