import boto3
import json
import requests

# boto3.setup_default_session(profile_name="quintodev")
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

MS_USERS_URL = "http://api.careassistant.co:8081"

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
    ciudad, especialidad  = extraer_ciudad_y_especialidad(pregunta)
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
    - No debes tener encuenta conversaciones previas. Solo debes responder a la pregunta actual.
    - No debes mencionar que eres un modelo de lenguaje o un asistente virtual. Tu función es ayudar a los usuarios a encontrar profesionales de salud.
    - Si la pregunta no trae no la información necesaria para responder, no entregues la respuesta con todos los profesionales.
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
