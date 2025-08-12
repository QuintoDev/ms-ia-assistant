from fastapi import FastAPI
from pydantic import BaseModel
from typing import Any, Dict, Optional, List
import os, json
import boto3

app = FastAPI(title="IA Service – Bedrock (paso a paso)")

# --- Config Bedrock ---
REGION_NAME = os.getenv("REGION_NAME", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
bedrock = boto3.client("bedrock-runtime", region_name=REGION_NAME)

class Ask(BaseModel):
    user_id: str
    step: str                     # dni | nombre_apellido | ooss | servicio | endoscopia_tipo | sede | datetime | done | servicios_info
    slots: Dict[str, Any] = {}    # snapshot de estado
    context: Dict[str, Any] = {}  # datos que pasa el backend (choices, sedes, servicios)
    reason: Optional[str] = None  # "greet" | "invalid" | "no_data" | None
    lang: str = "es"

SYSTEM_PROMPT = (
    "Eres un asistente de agendamiento médico que SOLO guía al usuario por un flujo ESTRICTO paso a paso. "
    "Nunca des información médica, diagnósticos ni tratamientos. "
    "Responde SIEMPRE en español, breve, claro y con tono amable. "
    "No inventes datos y usa SOLO la información de 'slots' y 'context'. "
    "No permitas saltar pasos: si 'reason' es 'invalid', repite con un ejemplo simple. "
    "Si 'reason' es 'greet', da la bienvenida y pide el dato del paso actual. "
    "Si 'step' es 'sede' y hay 'context.sedes', muestra esa lista sin agregar ni quitar. "
    "Si 'step' es 'datetime' y hay 'context.choices', crea un menú numerado 1..N y aclara que también aceptas letras A.. "
    "Si 'step' es 'servicios_info' y hay 'context.servicios', LISTA EXACTAMENTE esos ítems, sin agregar ni quitar. "
    "No menciones que eres IA ni otros sistemas. No pidas datos ya presentes en 'slots'. "
)

def _render_sedes(sedes: List[Dict[str, Any]]) -> str:
    lines = [f"- {x.get('sede_nombre','')} — {x.get('sede_direccion','')} [ID: {x.get('sede_id','')}]"
             for x in sedes[:8]]
    extra = f"\n… y {len(sedes)-8} más." if len(sedes) > 8 else ""
    return "\n".join(lines) + extra

def _render_choices(choices: List[str]) -> str:
    rows = [f"{i+1}) {c}" for i, c in enumerate(choices)]
    if not rows:
        return ""
    last_letter = chr(ord('A') + len(choices) - 1)
    rows.append(f"Responde con el número (1-{len(choices)}) o la letra (A-{last_letter}).")
    return "\n".join(rows)

def _render_services(svcs: List[str]) -> str:
    return "\n".join(f"- {s}" for s in svcs if s)

def build_user_prompt(a: Ask) -> str:
    # Compacta el snapshot para Bedrock
    sede_txt = ""
    if isinstance(a.slots.get("sede"), dict):
        s = a.slots["sede"]
        sede_txt = f"{s.get('sede_nombre','')} [ID {s.get('sede_id','')}], {s.get('sede_direccion','')}"
    slots_view = (
        f"DNI: {a.slots.get('dni') or '-'} | "
        f"Nombre: {a.slots.get('nombre_apellido') or '-'} | "
        f"OOSS: {a.slots.get('ooss') or '-'} | "
        f"Servicio: {a.slots.get('servicio') or '-'}"
        f"{' ('+a.slots.get('endoscopia_tipo')+')' if a.slots.get('servicio')=='ENDOSCOPIAS' and a.slots.get('endoscopia_tipo') else ''} | "
        f"Sede: {sede_txt or '-'} | "
        f"Fecha/Hora: {(a.slots.get('fecha') or '-') + ' ' + (a.slots.get('hora') or '-')}"
    )

    services_block = ""
    if a.step == "servicios_info" and isinstance(a.context.get("servicios"), list) and a.context["servicios"]:
        services_block = "Servicios disponibles (usa exactamente esta lista):\n" + _render_services(a.context["servicios"])

    sedes_block = ""
    if a.step == "sede" and isinstance(a.context.get("sedes"), list) and a.context["sedes"]:
        sedes_block = "Listado de sedes (usa exactamente esta lista):\n" + _render_sedes(a.context["sedes"])

    choices_block = ""
    if a.step == "datetime" and isinstance(a.context.get("choices"), list) and a.context["choices"]:
        choices_block = "Horarios disponibles (usa exactamente este menú):\n" + _render_choices(a.context["choices"])

    reason_hint = ""
    if a.reason == "greet":
        reason_hint = "Motivo: saludo inicial. Da la bienvenida y pide el dato del paso actual."
    elif a.reason == "invalid":
        reason_hint = "Motivo: entrada inválida. Repite la instrucción del paso con un ejemplo claro."
    elif a.reason == "no_data":
        reason_hint = "Motivo: no hay datos disponibles. Indica que no se pudo y sugiere intentar más tarde."

    # Instrucción específica por step
    step_instructions = {
        "servicios_info": "Responde listando los servicios disponibles en viñetas. No inicies el flujo ni pidas más datos.",
        "dni": "Pide DNI con ejemplo ('mi DNI es 12345678').",
        "nombre_apellido": "Pide nombre y apellido con ejemplo ('Me llamo Ana Pérez').",
        "ooss": "Pide obra social o 'Particular'.",
        "servicio": "Pide servicio entre: Consultas, SIBO o Endoscopias.",
        "endoscopia_tipo": "Pide elegir VEDA, VCC o Ambas.",
        "sede": "Muestra la lista de sedes proporcionada y pide elegir por ID o nombre.",
        "datetime": "Muestra el menú de horarios proporcionado y pide elegir (1..N o A..).",
        "done": "Muestra un resumen de todo y pide confirmación (sí/no).",
        "timeout": "La sesión se cerró por inactividad. Despídete cordialmente y explica cómo retomar: escribir 'hola' o reenviar su DNI para empezar de nuevo. No pidas más datos.",
    }.get(a.step, "Guía al siguiente paso brevemente.")

    return (
        f"Lenguaje: {a.lang}\n"
        f"Paso actual: {a.step}\n"
        f"{reason_hint}\n"
        f"Instrucción del paso: {step_instructions}\n"
        f"Estado (slots): {slots_view}\n\n"
        f"{services_block}\n\n"
        f"{sedes_block}\n\n"
        f"{choices_block}\n\n"
        "Redacta UNA sola respuesta breve, clara y amable, sin adornos innecesarios."
    )

def call_bedrock(prompt_text: str) -> str:
    body = {
        "system": [{"text": SYSTEM_PROMPT}],
        "messages": [{"role": "user", "content": [{"text": prompt_text}]}],
        "inferenceConfig": {
            "maxTokens": 220,
            "temperature": 0.1,  # menos creatividad para evitar inventar
            "topP": 0.9
        }
    }
    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    out = json.loads(resp["body"].read())
    return out["output"]["message"]["content"][0]["text"].strip()

@app.get("/health")
def health():
    return {"status": "UP"}

@app.post("/reply")
def reply(a: Ask):
    try:
        prompt = build_user_prompt(a)
        text = call_bedrock(prompt)
        # micro-sanitizer: evita que invente acciones externas
        text = text.replace("haz clic", "indica").replace("link", "opción")
        return {"text": text}
    except Exception as e:
        # Fallback mínimo si Bedrock falla
        # Para servicios_info, usa el contexto real si lo hay
        if a.step == "timeout":
            nombre = ""
            try:
                nombre = (a.slots or {}).get("nombre_apellido") or ""
            except:
                pass
            base = f"Hola{(' ' + nombre) if nombre else ''}. "
            return {"text": base + "Cerré tu sesión por inactividad. Para continuar, escribe “hola” o envía tu DNI para empezar de nuevo."}
        if a.step == "servicios_info" and isinstance(a.context, dict) and a.context.get("servicios"):
            svcs = a.context["servicios"]
            return {"text": "Servicios disponibles:\n" + "\n".join(f"- {s}" for s in svcs)}
        fallbacks = {
            "dni": "Para empezar, ¿tu DNI? Ej: “mi DNI es 12345678”.",
            "nombre_apellido": "¿Tu nombre y apellido? Ej: “Me llamo Ana Pérez”.",
            "ooss": "¿Cuál es tu obra social? Si no tienes, escribe “Particular”.",
            "servicio": "¿Qué servicio necesitas? Consultas, SIBO o Endoscopias.",
            "endoscopia_tipo": "Elige: VEDA, VCC o Ambas.",
            "sede": "Elige una sede por nombre o ID.",
            "datetime": "Elige un horario de la lista (1..N).",
            "done": "Tengo todo listo. ¿Confirmamos? (sí/no)",
        }
        return {"text": fallbacks.get(a.step, "¿Seguimos?")}
