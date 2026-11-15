"""Verificación NLI de cada afirmación frente al fragmento que cita (Bowman et al., 2015;
Williams et al., 2018), con un codificador multilingüe (Conneau et al., 2020).

``VerificadorLexico`` es un respaldo sin modelos, basado en el solapamiento de palabras de
contenido; solo sirve para pruebas y demostraciones sin red.
"""
import re

from .embeddings import normalizar

_VACIAS = set("""el la los las un una unos unas de del al y o u en a que se por para con sin sobre es son
 ser su sus lo le les como mas muy no si ya este esta esto estos estas ese esa eso""".split())


def _contenido(texto):
    return {p for p in re.findall(r"\w+", normalizar(texto)) if p not in _VACIAS and len(p) > 2}


class Verificador:
    def implicacion(self, premisa, hipotesis):
        """Probabilidad de que la premisa implique la hipótesis, en [0, 1]."""
        raise NotImplementedError


class VerificadorNLI(Verificador):
    """Clasificador NLI multilingüe de Hugging Face (``pip install transformers torch``)."""

    def __init__(self, modelo="joeddav/xlm-roberta-large-xnli"):
        import torch  # importación diferida
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(modelo)
        self.modelo = AutoModelForSequenceClassification.from_pretrained(modelo).eval()
        etiquetas = {v.lower(): k for k, v in self.modelo.config.id2label.items()}
        self.idx = etiquetas.get("entailment", etiquetas.get("implicación", 2))

    def implicacion(self, premisa, hipotesis):
        enc = self.tok(premisa, hipotesis, return_tensors="pt", truncation=True, max_length=512)
        with self.torch.no_grad():
            logits = self.modelo(**enc).logits[0]
        return float(self.torch.softmax(logits, dim=-1)[self.idx])


class VerificadorLexico(Verificador):
    """Respaldo: fracción de palabras de contenido de la hipótesis presentes en la premisa."""

    def implicacion(self, premisa, hipotesis):
        h = _contenido(hipotesis)
        if not h:
            return 1.0
        return len(h & _contenido(premisa)) / len(h)
