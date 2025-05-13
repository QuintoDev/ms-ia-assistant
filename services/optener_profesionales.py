import boto3
import json
import requests

boto3.setup_default_session(profile_name="quintodev")
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

MS_USERS_URL = "http://localhost:8080"

def obtener_profesionales(especialidad: str, ciudad: str):
    try:
        response = requests.get(
            f"{MS_USERS_URL}/users",
            params={"especialidad": especialidad, "ciudad": ciudad},
            timeout=5
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print("Error al obtener profesionales:", e)
        return []

def generar_contexto(profesionales: list) -> str:
    if not profesionales:
        return "Actualmente no hay profesionales disponibles para esta búsqueda."
    return "\n".join([
        f"- {p['nombre']} ({p['especialidad']}, {p['ciudad']})"
        for p in profesionales
    ])

def consultar_gpt_dinamico(pregunta: str, usuario: str, ciudad: str, especialidad: str) -> str:
    profesionales = obtener_profesionales(especialidad, ciudad)
    contexto = generar_contexto(profesionales)

    body = {
        "system": [
            {
                "text": f"Eres CareAssistant, un asistente de salud. Solo puedes responder con base en los profesionales disponibles. "
                        f"Usuario: {usuario}, ciudad: {ciudad}, especialidad requerida: {especialidad}.\n\n"
                        f"Profesionales disponibles:\n{contexto}"
            }
        ],
        "messages": [
            {
                "role": "user",
                "content": [{"text": pregunta}]
            }
        ],
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