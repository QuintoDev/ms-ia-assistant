import os
import boto3
import json
import requests

MS_ORCHESTRATOR_SERVICE = os.getenv("MS_ORCHESTRATOR_SERVICE", "http://localhost:8080")
REGION_NAME = os.getenv("REGION_NAME", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")

bedrock = boto3.client("bedrock-runtime", region_name=REGION_NAME)

def extraer_ciudad_y_especialidad(pregunta: str):
    body = {
        "system": [{
            "text": "Eres un asistente que ayuda a los usuarios a agendar: Consultas medicas, Examenes de SIBO (Small Intestinal Bacterial Overgrowth) y Endoscopias digestivas: la Videoendoscopia Digestiva Alta (VEDA) y la Videocolonoscopia (VCC"
        }],
        "messages": [{
            "role": "user",
            "content": [{"text": pregunta}]
        }],
        "inferenceConfig": {
            "maxTokens": 200,
            "temperature": 0.3
        }
    }

    response = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    text = result["output"]["message"]["content"][0]["text"]

    try:
        data = json.loads(text)
        return data.get("ciudad", ""), data.get("especialidad", "")
    except:
        return "", ""

def generar_contexto(profesionales: list) -> str:
    if not profesionales:
        return "Actualmente no hay profesionales disponibles para esta búsqueda."
    
    return "\n".join([
        f"- {p['nombre']} {p['apellido']} ({p['especialidad']} en {p['ciudad']}). Disponibilidad: {', '.join(p.get('disponibilidad', []))}. [ID: {p['id']}]"
        for p in profesionales
    ])

def consultar_gpt_dinamico(pregunta: str) -> str:

    system_text = f"""
    Eres CareAssistant, un asistente de salud digital confiable y respetuoso.
    Tu función es ayudar a los usuarios registrados a encontrar profesionales de salud disponibles, utilizando exclusivamente la información interna del sistema.

    Datos detectados:
    - Usuario: {usuario}
    - Ciudad: {ciudad or 'No detectada'}
    - Especialidad: {especialidad or 'No detectada'}

    Instrucciones:
    
    - Al finalizar, si el usuario desea agendar una cita o servicio con alguno de estos profesionales, invítalo a continuar el proceso directamente desde nuestra plataforma.
    - No debes proporcionar información médica, diagnósticos o tratamientos. Tu función es ayudar a los usuarios a encontrar profesionales de salud.
    - No inventes datos. Si no hay profesionales disponibles, indica que no se encontraron resultados en este momento.
    - No debes recomendar llamadas externas, recomendaciones de otros sistemas, redes sociales, paginas web, pasos o acciones fuera de la plataforma CareAssistant.
    - No debes tener encuenta conversaciones previas. Solo debes responder a la pregunta actual.
    - No debes mencionar que eres un modelo de lenguaje o un asistente virtual. Tu función es ayudar a los usuarios a encontrar profesionales de salud.
    - Si la pregunta no trae la información necesaria para responder, no entregues la respuesta con todos los profesionales.
    - Entrega siempre el mismo formato de respuesta para lista de profesionales disponibles".
    - No debes entregar lo que recibes de profesionales si no te lo piden explícitamente.
    - Formato ESTRICTO de respuesta para cada profesional (usa siempre este):
      - **Nombre del profesional** Especialidad. Ciudad. Disponibilidad: **Días**. [ID: uuid]. sobre el profesional.
    - Ejemplo:
      - **Lizeth Torres** Especialidad: Geriatria . Disponibilidad: **Lunes, Martes**. [ID: a9035aed-76e1-4632-80b9-f38c936f0964]. Soy una profesional de la salud con más de 10 años de experiencia en geriatría. Me apasiona ayudar a los adultos mayores a mantener su salud y bienestar. Estoy aquí para responder cualquier pregunta que tengas sobre el cuidado de la salud en esta etapa de la vida.
    - Asegúrate de seguir este formato para cada profesional listado.

    """

    body = {
        "system": [{"text": system_text.strip()}],
        "messages": [{"role": "user", "content": [{"text": pregunta}]}],
        "inferenceConfig": {
            "maxTokens": 300,
            "temperature": 0.7,
            "topP": 1.0
        }
    }

    response = bedrock.invoke_model(
        modelId="amazon.nova-micro-v1:0",
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )

    result = json.loads(response["body"].read())
    return result["output"]["message"]["content"][0]["text"]
