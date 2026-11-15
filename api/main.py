"""API REST del prototipo (FastAPI; Ramírez, 2018).

Orquesta los servicios, gestiona la autenticación y aplica las reglas de emergencia antes de
invocar cualquier modelo (la lógica está en ``api/orquestador.py``). Para ejecutarla::

    pip install -r requirements.txt
    export API_KEY=una-clave-larga      # obligatoria, salvo MODO_DESARROLLO=1
    export RAG_CORPUS=corpus/           # carpeta con los documentos aprobados (opcional)
    uvicorn api.main:app --reload

Variables de entorno: ``BD`` (ruta SQLite, por defecto ``prototipo.db``), ``POLITICA`` (por
defecto ``reglas_expertas``), ``RAG_CORPUS``, ``MODO_RAG`` (``produccion`` usa E5, ChromaDB y el
verificador NLI; ``prueba`` usa los respaldos sin red), ``VOZ=1`` para cargar Whisper y VITS.
La autenticación es una clave en la cabecera ``X-API-Key``, adecuada para un prototipo; un despliegue
real necesita un esquema de usuarios y tokens.
"""
import os
import secrets
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from .bd import BD
from .orquestador import Orquestador
from .servicios import ServicioAdaptativo


def _construir_rag(modo):
    from rag.ajustes import AjustesRAG
    from rag.asistente import AsistenteRAG
    from rag.corpus import cargar_corpus
    aj = AjustesRAG()
    if modo == "produccion":
        from rag.embeddings import EmbedderE5
        from rag.generador import GeneradorChat
        from rag.indice import IndiceChroma
        from rag.nli import VerificadorNLI
        asist = AsistenteRAG(EmbedderE5(aj.modelo_embeddings), IndiceChroma(), 
                             GeneradorChat(aj.llm_url, aj.llm_modelo, aj.llm_clave),
                             VerificadorNLI(aj.modelo_nli), aj)
    else:
        from rag.embeddings import EmbedderHash
        from rag.generador import GeneradorExtractivo
        from rag.indice import IndiceMemoria
        from rag.nli import VerificadorLexico
        asist = AsistenteRAG(EmbedderHash(), IndiceMemoria(), GeneradorExtractivo(), VerificadorLexico(), aj)
    corpus = os.getenv("RAG_CORPUS")
    if corpus and os.path.isdir(corpus):
        docs = cargar_corpus(corpus)
        version = asist.indexar(docs)
        return asist, version, [d.meta.get("id") for d in docs if d.aprobado]
    return asist, "", []


def crear_app():
    bd = BD(os.getenv("BD", "prototipo.db"))
    servicio = ServicioAdaptativo(bd, os.getenv("POLITICA", "reglas_expertas"))
    asistente, version, docs = _construir_rag(os.getenv("MODO_RAG", "prueba"))
    if version:
        bd.registrar_version_corpus(version, docs)
    transcriptor = sintetizador = None
    if os.getenv("VOZ") == "1":
        from voz.stt import Transcriptor
        from voz.tts import Sintetizador
        transcriptor, sintetizador = Transcriptor(), Sintetizador()
    orq = Orquestador(servicio, asistente, transcriptor, sintetizador)

    app = FastAPI(title="Asistente de autocuidado y turismo rural - ANUC Villeta", version="1.0")

    def autenticar(x_api_key: Optional[str] = Header(default=None)):
        clave = os.getenv("API_KEY", "")
        if not clave and os.getenv("MODO_DESARROLLO") == "1":
            return
        if not clave or not x_api_key or not secrets.compare_digest(x_api_key, clave):
            raise HTTPException(status_code=401, detail="Clave de API inválida")

    class Registro(BaseModel):
        labor: str
        edad: str
        lectura: int = Field(ge=1, le=3)
        voz: bool = False
        diagnostico: Optional[List[float]] = Field(default=None, min_length=14, max_length=14)
        consiente_sensibles: bool = False
        condiciones: Optional[dict] = None

    class Respuesta(BaseModel):
        accion: int
        correcta: Optional[bool] = None
        completada: bool = True

    class Consulta(BaseModel):
        texto: str = Field(min_length=1, max_length=1000)

    class ConsultaVoz(BaseModel):
        confirmada: str = Field(min_length=1, max_length=1000)

    class PedidoRuta(BaseModel):
        coords: List[List[float]]
        puntuacion: List[float]
        duracion: List[float]
        ventana: List[List[float]]
        t_max: float
        inseguros: List[List[int]] = []

    @app.get("/salud")
    def salud():
        return {"estado": "ok"}

    @app.post("/personas", dependencies=[Depends(autenticar)])
    def registrar(r: Registro):
        return orq.registrar(**r.model_dump())

    @app.get("/personas/{codigo}/actividad", dependencies=[Depends(autenticar)])
    def actividad(codigo: str):
        try:
            return orq.siguiente_actividad(codigo)
        except KeyError:
            raise HTTPException(404, "Persona no registrada")

    @app.post("/personas/{codigo}/respuesta", dependencies=[Depends(autenticar)])
    def respuesta(codigo: str, r: Respuesta):
        try:
            return orq.registrar_respuesta(codigo, r.accion, r.correcta, r.completada)
        except (KeyError, TypeError):
            raise HTTPException(404, "Persona no registrada")

    @app.delete("/personas/{codigo}", dependencies=[Depends(autenticar)])
    def suprimir(codigo: str):
        bd.borrar_persona(codigo)
        return {"eliminada": codigo}

    @app.post("/asistente/consulta", dependencies=[Depends(autenticar)])
    def consulta(c: Consulta):
        return orq.consultar(c.texto)

    @app.post("/voz/transcribir", dependencies=[Depends(autenticar)])
    async def transcribir(audio: UploadFile = File(...)):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            tmp.write(await audio.read())
            tmp.flush()
            try:
                return orq.consultar_por_voz(tmp.name)
            except RuntimeError as e:
                raise HTTPException(503, str(e))

    @app.post("/voz/consulta", dependencies=[Depends(autenticar)])
    def consulta_voz(c: ConsultaVoz):
        return orq.consultar_por_voz(None, confirmada=c.confirmada)

    @app.post("/rutas", dependencies=[Depends(autenticar)])
    def ruta(p: PedidoRuta):
        return orq.ruta(p.coords, p.puntuacion, p.duracion, p.ventana, p.t_max, p.inseguros)

    return app


app = crear_app()
