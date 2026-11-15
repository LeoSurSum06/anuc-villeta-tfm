"""Parámetros del asistente RAG (Subsección 4.2.7). Se pueden sustituir con variables de entorno."""
import os
from dataclasses import dataclass


@dataclass
class AjustesRAG:
    # Fragmentación: entre 300 y 500 palabras con solapamiento de 50
    palabras_min: int = 300
    palabras_max: int = 500
    solapamiento: int = 50
    # Recuperación: los cinco fragmentos más similares
    k: int = 5
    # Umbral de abstención tau: si la similitud máxima es menor, el sistema se abstiene.
    # Debe calibrarse con preguntas de desarrollo distintas del banco de pruebas
    # (``calibrar_umbral`` en rag/asistente.py). El valor por defecto es solo de arranque.
    umbral_abstencion: float = float(os.getenv("RAG_UMBRAL", "0.80"))
    # Verificación NLI: una afirmación queda respaldada si P(implicación) >= este valor
    umbral_nli: float = float(os.getenv("RAG_UMBRAL_NLI", "0.50"))
    # Modelos intercambiables (se descargan la primera vez que se usan)
    modelo_embeddings: str = os.getenv("RAG_EMBEDDINGS", "intfloat/multilingual-e5-base")
    modelo_nli: str = os.getenv("RAG_NLI", "joeddav/xlm-roberta-large-xnli")
    # LLM: cualquier servicio compatible con la API de chat de OpenAI (local o externo)
    llm_url: str = os.getenv("LLM_URL", "http://localhost:11434/v1")
    llm_modelo: str = os.getenv("LLM_MODELO", "llama3")
    llm_clave: str = os.getenv("LLM_CLAVE", "")
    linea_emergencia: str = "123"
