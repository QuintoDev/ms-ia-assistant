import boto3
import json
import requests

# boto3.setup_default_session(profile_name="quintodev")
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

MS_USERS_URL = "http://localhost:8081"

def extraer_ciudad_y_especialidad(pregunta: str):
    body = {
        "system": [{
            "text": "Eres un asistente que extrae información de preguntas sobre salud. Extrae ciudad y especialidad médica mencionadas. Solo responde con un JSON así: {\"ciudad\": \"bogota\", \"especialidad\": \"geriatria\"}."
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
        modelId="amazon.nova-micro-v1:0",
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

def obtener_profesionales(especialidad: str, ciudad: str, token: str) -> list:
    try:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        response = requests.get(
            f"{MS_USERS_URL}/searches",
            params={"especialidad": especialidad, "ciudad": ciudad},
            timeout=5,
            headers=headers
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
        f"- {p['nombre']} {p['apellido']} ({p['especialidad']} en {p['ciudad']}). Disponibilidad: {', '.join(p.get('disponibilidad', []))}. [ID: {p['id']}]"
        for p in profesionales
    ])

def consultar_gpt_dinamico(pregunta: str, usuario: str, token: str) -> str:
    ciudad, especialidad = extraer_ciudad_y_especialidad(pregunta)
    profesionales = obtener_profesionales(especialidad, ciudad, token)
    contexto = generar_contexto(profesionales)

    system_text = f"""
    Eres es CareAssistant, un asistente de salud digital confiable y respetuoso.
    Tu función es ayudar a los usuarios registrados a encontrar profesionales de salud disponibles, utilizando exclusivamente la información interna del sistema.

    Datos detectados:
    - Usuario: {usuario}
    - Ciudad: {ciudad or 'No detectada'}
    - Especialidad: {especialidad or 'No detectada'}

    Lista de profesionales disponibles:
    {contexto}

    Instrucciones:
    
    - Al finalizar, si el usuario desea agendar una cita o servicio con alguno de estos profesionales, invítalo a continuar el proceso directamente desde nuestra plataforma.
    - No debes proporcionar información médica, diagnósticos o tratamientos. Tu función es ayudar a los usuarios a encontrar profesionales de salud.
    - No inventes datos. Si no hay profesionales disponibles, indica que no se encontraron resultados en este momento.
    - No debes recomendar llamadas externas, recomendaciones de otros sistemas, redes sociales, paginas web, pasos o acciones fuera de la plataforma CareAssistant.
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
