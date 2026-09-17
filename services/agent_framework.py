import inspect
import json
import os
import sys
from typing import Any, Callable, Dict, List, Optional
from dotenv import load_dotenv
from openai import OpenAI


def get_llm_client() -> OpenAI:
    """Inicializa y retorna el cliente OpenAI compatible con la configuración del .env."""
    load_dotenv()
    nvidia_key = os.getenv("NVIDIA_API_KEY")
    if nvidia_key:
        api_key = nvidia_key.strip().strip('"').strip("'")
        base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").strip()
        return OpenAI(api_key=api_key, base_url=base_url)

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        api_key = openai_key.strip().strip('"').strip("'")
        base_url = os.getenv("OPENAI_BASE_URL")
        return OpenAI(api_key=api_key, base_url=base_url.strip() if base_url else None)

    print("\n[ERROR] Falta NVIDIA_API_KEY u OPENAI_API_KEY en el archivo .env.")
    sys.exit(1)


def get_model_name() -> str:
    """Retorna el nombre del modelo configurado."""
    load_dotenv()
    if os.getenv("NVIDIA_API_KEY"):
        return os.getenv("NVIDIA_MODEL", "moonshotai/kimi-k3")
    return os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class AgentTool:
    """Envuelve una función de Python o un sub-agente para ser invocado como herramienta por un LLM."""
    def __init__(self, name: str, description: str, func: Callable, schema: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.func = func
        self.schema = schema or self._build_schema()

    def _build_schema(self) -> Dict[str, Any]:
        sig = inspect.signature(self.func)
        properties = {}
        required = []

        for p_name, param in sig.parameters.items():
            if p_name in ("self", "args", "kwargs"):
                continue
            properties[p_name] = {
                "type": "string",
                "description": f"Parámetro {p_name}"
            }
            if param.default == inspect.Parameter.empty:
                required.append(p_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        }

    def execute(self, **kwargs) -> str:
        res = self.func(**kwargs)
        if isinstance(res, str):
            return res
        return json.dumps(res, ensure_ascii=False)


class Agent:
    """
    Clase Agent para orquestación multiagente que soporta:
    - as_tool(): convierte un agente en herramienta para un supervisor o manager.
    - handoffs=[agente1, agente2]: permite transferir la conversación entre agentes pares.
    """
    def __init__(
        self,
        name: str,
        instructions: str,
        tools: Optional[List[Any]] = None,
        handoffs: Optional[List["Agent"]] = None
    ):
        self.name = name
        self.instructions = instructions
        self.tools: Dict[str, AgentTool] = {}
        self.handoff_targets: Dict[str, "Agent"] = {}

        # 1. Registrar herramientas estándar o sub-agentes (as_tool)
        if tools:
            for t in tools:
                if isinstance(t, AgentTool):
                    self.tools[t.name] = t
                elif callable(t):
                    doc = t.__doc__ or f"Ejecuta la función {t.__name__}"
                    tool_obj = AgentTool(
                        name=t.__name__,
                        description=doc.strip().split('\n')[0],
                        func=t
                    )
                    self.tools[tool_obj.name] = tool_obj

        # 2. Registrar handoffs a partir de la lista de agentes 
        if handoffs:
            self.set_handoffs(handoffs)

    def set_handoffs(self, handoff_agents: List["Agent"]) -> None:
        """Configura la lista de agentes a los cuales este agente puede transferir el control."""
        self.handoff_targets.clear()
        for target in handoff_agents:
            clean_name = target.name.lower().replace(" ", "_").replace("-", "_")
            tool_name = f"transfer_to_{clean_name}"
            self.handoff_targets[tool_name] = target

    def as_tool(self, tool_name: Optional[str] = None, tool_description: Optional[str] = None) -> AgentTool:
        """
        Convierte este Agente en una herramienta (as_tool) para un Supervisor o Manager.
        Sigue la estructura:
            agent.as_tool(tool_name="...", tool_description="...")
        """
        clean_name = self.name.lower().replace(" ", "_").replace("-", "_")
        name = tool_name or f"{clean_name}_expert"
        desc = tool_description or f"Delega una tarea al agente especialista {self.name}."

        schema = {
            "type": "function",
            "function": {
                "name": name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "instruccion": {
                            "type": "string",
                            "description": f"Instrucción detallada o requerimiento para {self.name}."
                        }
                    },
                    "required": ["instruccion"]
                }
            }
        }

        # Sub-bucle de ejecución cuando el supervisor/manager llama a este agente como herramienta
        def run_subagent_task(instruccion: str) -> str:
            sub_messages = [
                {"role": "system", "content": self.instructions},
                {"role": "user", "content": instruccion}
            ]
            client = get_llm_client()
            model = get_model_name()
            tools_schemas = [t.schema for t in self.tools.values()]

            for _ in range(5):
                resp = client.chat.completions.create(
                    model=model,
                    messages=sub_messages,
                    tools=tools_schemas if tools_schemas else None,
                    tool_choice="auto" if tools_schemas else None
                )
                choice = resp.choices[0]
                msg = choice.message

                if msg.tool_calls:
                    sub_messages.append(msg)
                    for tc in msg.tool_calls:
                        tname = tc.function.name
                        try:
                            targs = json.loads(tc.function.arguments)
                        except Exception:
                            targs = {}

                        print(f"    [{self.name}] -> Invocando '{tname}'")
                        if tname in self.tools:
                            res_str = self.tools[tname].execute(**targs)
                        else:
                            res_str = f"Error: Tool '{tname}' desconocida."

                        sub_messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tname,
                            "content": res_str
                        })
                else:
                    return msg.content or "Tarea completada."

            return "Límite de pasos alcanzado en sub-agente."

        return AgentTool(name=name, description=desc, func=run_subagent_task, schema=schema)

    def _get_active_tools_schemas(self) -> List[Dict[str, Any]]:
        schemas = [t.schema for t in self.tools.values()]

        # Generar automáticamente schemas para cada agente en handoff_targets
        for tool_name, target in self.handoff_targets.items():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"Transfiere el control de la conversación al agente {target.name}.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "motivo": {"type": "string", "description": "Motivo de la transferencia."}
                        }
                    }
                }
            })
        return schemas

    def run_turn(
        self,
        messages: List[Dict[str, Any]],
        client: OpenAI,
        model: str
    ) -> Dict[str, Any]:
        """
        Ejecuta el turno conversacional para este agente.
        Devuelve {'content': str, 'handoff_to': Optional[Agent]}.
        """
        full_messages = [{"role": "system", "content": self.instructions}] + messages
        tools_schemas = self._get_active_tools_schemas()

        for _ in range(5):
            resp = client.chat.completions.create(
                model=model,
                messages=full_messages,
                tools=tools_schemas if tools_schemas else None,
                tool_choice="auto" if tools_schemas else None
            )
            msg = resp.choices[0].message

            if msg.tool_calls:
                full_messages.append(msg)
                messages.append(msg)

                for tc in msg.tool_calls:
                    tname = tc.function.name
                    try:
                        targs = json.loads(tc.function.arguments)
                    except Exception:
                        targs = {}

                    # 1. Comprobar si es un HANDOFF
                    if tname in self.handoff_targets:
                        target = self.handoff_targets[tname]
                        print(f"  [HANDOFF] {self.name} -> {target.name}")
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "name": tname,
                            "content": json.dumps({"status": "transferencia_exitosa", "agente": target.name})
                        })
                        return {"content": None, "handoff_to": target}

                    # 2. Ejecutar herramienta estándar o sub-agente (as_tool)
                    print(f"  [{self.name}] -> Invocando '{tname}'")
                    if tname in self.tools:
                        result = self.tools[tname].execute(**targs)
                    else:
                        result = f"Error: Tool '{tname}' desconocida."

                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "name": tname,
                        "content": result
                    }
                    full_messages.append(tool_msg)
                    messages.append(tool_msg)
            else:
                messages.append({"role": "assistant", "content": msg.content})
                return {"content": msg.content, "handoff_to": None}

        return {"content": "Límite de turnos alcanzado.", "handoff_to": None}
