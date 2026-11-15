"""Asistente RAG con filtros previos, abstención y verificación (Figura 4.3).

Seis etapas: (1) filtro de señales de alarma, (2) filtro de alcance clínico, (3) recuperación,
(4) umbral de abstención, (5) generación restringida y (6) verificación NLI. Las etapas 1 y 2
se ejecutan fuera del modelo de lenguaje.
"""
import re
from dataclasses import dataclass, field

import numpy as np

from .ajustes import AjustesRAG
from .corpus import fragmentos_aprobados
from .filtros import (MENSAJE_ABSTENCION, MENSAJE_DERIVACION, MENSAJE_EMERGENCIA,
                      detectar_alarma, pide_alcance_clinico)


@dataclass
class Respuesta:
    tipo: str                       # emergencia | derivacion | abstencion | respuesta
    texto: str
    citas: list = field(default_factory=list)     # [{n, documento, seccion, version}]
    similitud_max: float = 0.0
    afirmaciones_eliminadas: list = field(default_factory=list)
    version_indice: str = ""


def _dividir_afirmaciones(texto):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", texto.strip()) if len(s.strip()) > 3]


class AsistenteRAG:
    def __init__(self, embedder, indice, generador, verificador, ajustes=None):
        self.embedder, self.indice = embedder, indice
        self.generador, self.verificador = generador, verificador
        self.aj = ajustes or AjustesRAG()
        self.version_indice = ""

    # --------------------------------------------------------------- indexación
    def indexar(self, documentos):
        """Indexa solo los documentos aprobados y devuelve la versión del índice."""
        frags, version = fragmentos_aprobados(
            documentos, palabras_min=self.aj.palabras_min, palabras_max=self.aj.palabras_max,
            solapamiento=self.aj.solapamiento)
        self.indice.vaciar()
        if frags:
            self.indice.agregar(frags, self.embedder.embed_documentos([f.texto for f in frags]))
        self.version_indice = version
        return version

    # ------------------------------------------------------------------ consulta
    def responder(self, consulta):
        aj = self.aj
        # 1. Filtro de señales de alarma (antes de cualquier otra operación)
        if detectar_alarma(consulta):
            return Respuesta("emergencia", MENSAJE_EMERGENCIA.format(linea=aj.linea_emergencia),
                             version_indice=self.version_indice)
        # 2. Filtro de alcance clínico
        if pide_alcance_clinico(consulta):
            return Respuesta("derivacion", MENSAJE_DERIVACION.format(linea=aj.linea_emergencia),
                             version_indice=self.version_indice)
        # 3. Recuperación de los k fragmentos más similares
        hallazgos = self.indice.buscar(self.embedder.embed_consulta(consulta), aj.k)
        sim_max = max((s for _, s in hallazgos), default=0.0)
        # 4. Umbral de abstención
        if not hallazgos or sim_max < aj.umbral_abstencion:
            return Respuesta("abstencion", MENSAJE_ABSTENCION, similitud_max=sim_max,
                             version_indice=self.version_indice)
        frags = [f for f, _ in hallazgos]
        # 5. Generación restringida a los fragmentos
        borrador = self.generador.generar(consulta, frags)
        if "NO_HAY_INFORMACION" in borrador:
            return Respuesta("abstencion", MENSAJE_ABSTENCION, similitud_max=sim_max,
                             version_indice=self.version_indice)
        # 6. Verificación NLI de cada afirmación frente al fragmento citado
        afirmaciones = _dividir_afirmaciones(borrador)
        conservadas, eliminadas, usadas = [], [], set()
        for i, af in enumerate(afirmaciones):
            citadas = [int(n) for n in re.findall(r"\[(\d+)\]", af) if 1 <= int(n) <= len(frags)]
            limpia = re.sub(r"\s*\[\d+\]", "", af).strip()
            apoyada = bool(citadas) and any(
                self.verificador.implicacion(frags[n - 1].texto, limpia) >= aj.umbral_nli for n in citadas)
            if apoyada:
                conservadas.append(af)
                usadas.update(citadas)
            else:
                eliminadas.append(af)
                if i == 0:      # la afirmación central (la primera) no queda respaldada: abstención
                    return Respuesta("abstencion", MENSAJE_ABSTENCION, similitud_max=sim_max,
                                     afirmaciones_eliminadas=eliminadas, version_indice=self.version_indice)
        if not conservadas:
            return Respuesta("abstencion", MENSAJE_ABSTENCION, similitud_max=sim_max,
                             afirmaciones_eliminadas=eliminadas, version_indice=self.version_indice)
        citas = [{"n": n, "documento": frags[n - 1].documento, "seccion": frags[n - 1].seccion,
                  "version": frags[n - 1].version} for n in sorted(usadas)]
        return Respuesta("respuesta", " ".join(conservadas), citas, sim_max, eliminadas, self.version_indice)


def calibrar_umbral(asistente, preguntas_desarrollo, cobertura=0.9):
    """Calibra tau con preguntas de desarrollo DISTINTAS del banco de pruebas.

    ``preguntas_desarrollo`` es una lista de (pregunta, respondible: bool). Devuelve el
    umbral que conserva la ``cobertura`` de las preguntas respondibles (cuantil inferior de su
    similitud máxima) y, de paso, la tasa de preguntas no respondibles que ese umbral deja pasar.
    """
    sims_si, sims_no = [], []
    for q, respondible in preguntas_desarrollo:
        h = asistente.indice.buscar(asistente.embedder.embed_consulta(q), asistente.aj.k)
        s = max((x for _, x in h), default=0.0)
        (sims_si if respondible else sims_no).append(s)
    tau = float(np.quantile(sims_si, 1 - cobertura)) if sims_si else asistente.aj.umbral_abstencion
    falsos = float(np.mean([s >= tau for s in sims_no])) if sims_no else float("nan")
    return tau, falsos
