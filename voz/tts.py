"""Síntesis de voz con un modelo ligero de tipo VITS (Kim et al., 2021).

Usa Coqui TTS (``pip install TTS``) con un modelo VITS en español. Los audios de las microlecciones
se sintetizan de antemano y se revisan, de modo que el contenido formativo nunca depende de una
síntesis en tiempo real; solo las respuestas del asistente se sintetizan bajo demanda.
"""
from pathlib import Path

MODELO_ES = "tts_models/es/css10/vits"


class Sintetizador:
    def __init__(self, modelo=MODELO_ES):
        from TTS.api import TTS  # importación diferida
        self.tts = TTS(modelo)

    def sintetizar(self, texto, ruta_salida):
        """Escribe un archivo WAV con el texto leído y devuelve su ruta."""
        Path(ruta_salida).parent.mkdir(parents=True, exist_ok=True)
        self.tts.tts_to_file(text=texto, file_path=str(ruta_salida))
        return str(ruta_salida)


def pre_sintetizar_microlecciones(sintetizador, directorio_textos, directorio_audios):
    """Sintetiza de antemano todos los textos aprobados (.txt o .md) de un directorio.

    Devuelve la lista de archivos generados. Cada audio debe revisarse antes de publicarse.
    """
    generados = []
    for texto in sorted(Path(directorio_textos).glob("*")):
        if texto.suffix.lower() in (".txt", ".md"):
            salida = Path(directorio_audios) / (texto.stem + ".wav")
            if not salida.exists():
                sintetizador.sintetizar(texto.read_text(encoding="utf8"), salida)
            generados.append(str(salida))
    return generados
