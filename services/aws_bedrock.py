# app.py — API mínima con saludo vía Bedrock

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os, json
import boto3

# --- Config ---
REGION_NAME = os.getenv("REGION_NAME", "us-east-1")
MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "amazon.nova-micro-v1:0")
bedrock = boto3.client("bedrock-runtime", region_name=REGION_NAME)

app = FastAPI(title="Agenda Médica – Paso 1 (Saludo)")

class Mensaje(BaseModel):
    usuario: str

def saludar(usuario: str) -> str:
    system_text = "Eres un asistente amable que saluda cordialmente a los usuarios de un servicio de agendamiento médico. Responde en una sola frase."
    body = {
        "system": [{"text": system_text}],
        "messages": [{"role": "user", "content": [{"text": f"Saluda al usuario {usuario}"}]}],
        "inferenceConfig": {"maxTokens": 50, "temperature": 0.7}
    }

    resp = bedrock.invoke_model(
        modelId=MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps(body)
    )
    result = json.loads(resp["body"].read())
    return result["output"]["message"]["content"][0]["text"]

@app.get("/")
def root():
    return {"status": "UP"}

@app.post("/saludo")
def saludo(msg: Mensaje):
    try:
        texto = saludar(msg.usuario)
        return {"respuesta": texto}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
