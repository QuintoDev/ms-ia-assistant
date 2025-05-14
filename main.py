from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import Header
from services.aws_bedrock import consultar_gpt_dinamico

app = FastAPI()

class ConsultaIA(BaseModel):
    pregunta: str
    usuario: str

@app.post("/conversations")
def consultar_ia(data: ConsultaIA, authorization: str = Header(...)):
    try:
        token = authorization.replace("Bearer ", "")
        respuesta = consultar_gpt_dinamico(
            pregunta=data.pregunta,
            usuario=data.usuario,
            token=token
        )
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))