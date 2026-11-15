"""Capa de orquestación: todo el tráfico de la persona pasa por aquí.

Centraliza las reglas de emergencia (que se aplican antes de invocar cualquier modelo), el registro
de interacciones y la composición de los servicios adaptativo, RAG, rutas y voz. No depende de
FastAPI, de modo que puede probarse sin servidor; ``api/main.py`` solo expone sus métodos por HTTP.
"""
import dataclasses

from rag.filtros import MENSAJE_EMERGENCIA, detectar_alarma
from rutas.optw import Instancia, puntuacion_ruta, resolver_heuristica

import numpy as np


class Orquestador:
    def __init__(self, servicio_adaptativo, asistente=None, transcriptor=None, sintetizador=None,
                 umbral_confianza_voz=0.6):
        self.adaptativo = servicio_adaptativo
        self.asistente = asistente
        self.transcriptor = transcriptor
        self.sintetizador = sintetizador
        self.umbral_confianza_voz = umbral_confianza_voz

    # --------------------------------------------------------------- formación
    def registrar(self, **datos):
        return {"codigo": self.adaptativo.registrar(**datos)}

    def siguiente_actividad(self, codigo):
        return self.adaptativo.siguiente_actividad(codigo)

    def registrar_respuesta(self, codigo, accion, correcta, completada=True):
        return self.adaptativo.registrar_respuesta(codigo, accion, correcta, completada)

    # --------------------------------------------------------------- asistente
    def consultar(self, texto):
        """Consulta de texto. Las señales de alarma se atienden antes que cualquier modelo."""
        if detectar_alarma(texto):
            linea = self.asistente.aj.linea_emergencia if self.asistente else "123"
            return {"tipo": "emergencia", "texto": MENSAJE_EMERGENCIA.format(linea=linea), "citas": []}
        if self.asistente is None:
            return {"tipo": "abstencion", "texto": "El asistente documental no está configurado.", "citas": []}
        return dataclasses.asdict(self.asistente.responder(texto))

    def consultar_por_voz(self, audio, confirmada=None):
        """Voz: transcribe y devuelve la transcripción para que la persona la confirme o corrija.

        Si ``confirmada`` trae el texto confirmado, se ejecuta la consulta con él. Una
        transcripción nunca se ejecuta sin confirmación, para reducir el impacto de errores de
        reconocimiento; como excepción, si contiene una señal de alarma se responde de inmediato.
        """
        if confirmada is not None:
            resp = self.consultar(confirmada)
            if self.sintetizador is not None:
                resp["audio"] = self.sintetizador.sintetizar(resp["texto"], "salida/respuesta.wav")
            return resp
        if self.transcriptor is None:
            raise RuntimeError("El servicio de voz no está configurado.")
        texto, conf = self.transcriptor.transcribir(audio)
        if detectar_alarma(texto):
            return {"transcripcion": texto, "confianza": conf, **self.consultar(texto)}
        return {"transcripcion": texto, "confianza": conf, "requiere_confirmacion": True,
                "confianza_baja": conf < self.umbral_confianza_voz}

    # ------------------------------------------------------------------- rutas
    def ruta(self, coords, puntuacion, duracion, ventana, t_max, inseguros=()):
        """Recomienda una ruta para una finca. ``coords``/``puntuacion``/``duracion``/``ventana``
        incluyen en las posiciones 0 y n+1 la entrada y la salida (puntuación y duración 0)."""
        inst = Instancia(np.asarray(coords, float), np.asarray(puntuacion, float),
                         np.asarray(duracion, float), np.asarray(ventana, float), float(t_max),
                         inseguros={tuple(x) for x in inseguros})
        ruta, p = resolver_heuristica(inst)
        return {"ruta": [int(i) for i in ruta], "puntuacion": p, "puntos": len(ruta) - 2}
