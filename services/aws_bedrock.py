import boto3
import json

# Cliente de Amazon Bedrock Runtime
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def consultar_gpt(pregunta: str) -> str:
    body = {
        "system": [{"text": "Eres CareAssistant, un asistente de salud. Solo puedes sugerir profesionales registrados en nuestra base según especialidad médica, ciudad y disponibilidad. Usa el contexto del paciente para filtrar resultados útiles."}],
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
