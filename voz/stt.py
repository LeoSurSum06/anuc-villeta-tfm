"""Reconocimiento de voz con Whisper (Radford et al., 2023).

Usa ``faster-whisper`` (``pip install faster-whisper``), que ejecuta los pesos de Whisper de forma
eficiente en CPU. El sistema muestra y lee la transcripción antes de ejecutar una consulta y
permite corregirla; por eso ``transcribir`` devuelve el texto y una confianza aproximada.
También incluye la tasa de error por palabra (WER), que se medirá en el piloto de campo.
"""
import math
import re
import unicodedata


class Transcriptor:
    def __init__(self, modelo="small", dispositivo="cpu", tipo_calculo="int8"):
        from faster_whisper import WhisperModel  # importación diferida
        self.modelo = WhisperModel(modelo, device=dispositivo, compute_type=tipo_calculo)

    def transcribir(self, audio, idioma="es"):
        """Transcribe un archivo de audio (ruta o archivo binario).

        Devuelve ``(texto, confianza)``; la confianza es la probabilidad media de las palabras,
        en [0, 1], y sirve para pedir confirmación cuando es baja.
        """
        segmentos, _ = self.modelo.transcribe(audio, language=idioma, vad_filter=True, word_timestamps=True)
        palabras, probs = [], []
        for seg in segmentos:
            for w in (seg.words or []):
                palabras.append(w.word)
                probs.append(w.probability)
        texto = "".join(palabras).strip()
        conf = float(sum(probs) / len(probs)) if probs else 0.0
        return texto, conf


def _normalizar(t):
    t = unicodedata.normalize("NFD", t.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")
    return re.findall(r"\w+", t)


def wer(referencia, hipotesis):
    """Tasa de error por palabra: (sustituciones + inserciones + borrados) / palabras de la referencia."""
    r, h = _normalizar(referencia), _normalizar(hipotesis)
    if not r:
        return 0.0 if not h else math.inf
    d = list(range(len(h) + 1))
    for i, pr in enumerate(r, 1):
        prev, d[0] = d[0], i
        for j, ph in enumerate(h, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (pr != ph))
            prev, d[j] = d[j], cur
    return d[len(h)] / len(r)
