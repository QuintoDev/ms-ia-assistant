from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from services.aws_bedrock import consultar_gpt

app = FastAPI()

class ConsultaIA(BaseModel):
    pregunta: str

@app.post("/ia")
def consultar_ia(data: ConsultaIA):
    try:
        respuesta = consultar_gpt(data.pregunta)
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
