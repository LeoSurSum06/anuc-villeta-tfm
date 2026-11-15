"""Modelos de embeddings. El de producción es multilingual-E5 (Wang et al., 2024).

``EmbedderHash`` es un respaldo determinista, sin dependencias ni descargas, pensado solo para
pruebas y demostraciones sin red; no debe usarse para evaluar el asistente.
"""
import hashlib
import re
import unicodedata

import numpy as np


def normalizar(texto):
    """Minúsculas y sin tildes."""
    t = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in t if unicodedata.category(c) != "Mn")


class Embedder:
    def embed_documentos(self, textos):
        raise NotImplementedError

    def embed_consulta(self, texto):
        raise NotImplementedError


class EmbedderE5(Embedder):
    """multilingual-E5 con sentence-transformers (``pip install sentence-transformers``)."""

    def __init__(self, modelo="intfloat/multilingual-e5-base"):
        from sentence_transformers import SentenceTransformer  # importación diferida
        self.modelo = SentenceTransformer(modelo)

    def embed_documentos(self, textos):
        v = self.modelo.encode([f"passage: {t}" for t in textos], normalize_embeddings=True,
                               show_progress_bar=False)
        return np.asarray(v, dtype=np.float32)

    def embed_consulta(self, texto):
        v = self.modelo.encode([f"query: {texto}"], normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(v[0], dtype=np.float32)


class EmbedderHash(Embedder):
    """Respaldo: n-gramas de palabras y de caracteres proyectados por hashing (dimensión 1024)."""

    def __init__(self, dim=1024):
        self.dim = dim

    def _vec(self, texto):
        t = normalizar(texto)
        palabras = re.findall(r"\w+", t)
        rasgos = palabras + [" ".join(p) for p in zip(palabras, palabras[1:])]
        rasgos += [w[i:i + 4] for w in palabras for i in range(max(len(w) - 3, 1))]
        v = np.zeros(self.dim, dtype=np.float32)
        for r in rasgos:
            h = int.from_bytes(hashlib.md5(r.encode()).digest()[:8], "little")
            v[h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed_documentos(self, textos):
        return np.vstack([self._vec(t) for t in textos]) if textos else np.zeros((0, self.dim), np.float32)

    def embed_consulta(self, texto):
        return self._vec(texto)
