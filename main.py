from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi import Header
from fastapi.middleware.cors import CORSMiddleware
from services.aws_bedrock import consultar_gpt_dinamico

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

class ConsultaIA(BaseModel):
    pregunta: str
    usuario: str

@app.get("/")
def root():
    return {"status": "UP"}

@app.post("/conversations")
def consultar_ia(data: ConsultaIA, authorization: str = Header(...)):
    try:
        token = authorization.replace("Bearer ", "")
        respuesta = consultar_gpt_dinamico(
            pregunta=data.pregunta,
            usuario=data.usuario,
            token=token
        )
        print("pregunta:", data.pregunta)
        print("respuesta:", respuesta)
        return {"respuesta": respuesta}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))