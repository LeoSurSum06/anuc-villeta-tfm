"""Índices vectoriales. En producción, ChromaDB con índice HNSW (Malkov y Yashunin, 2020).

``IndiceMemoria`` es un índice exacto por producto escalar (los vectores están normalizados) que
sirve para pruebas y para corpus pequeños; ``IndiceChroma`` usa la base vectorial embebida.
"""
import json
from pathlib import Path

import numpy as np


class Indice:
    def agregar(self, fragmentos, vectores):
        raise NotImplementedError

    def buscar(self, vector, k=5):
        """Devuelve [(fragmento, similitud)] en orden decreciente de similitud."""
        raise NotImplementedError

    def vaciar(self):
        raise NotImplementedError


class IndiceMemoria(Indice):
    def __init__(self):
        self.fragmentos, self.matriz = [], None

    def agregar(self, fragmentos, vectores):
        self.fragmentos.extend(fragmentos)
        v = np.asarray(vectores, dtype=np.float32)
        self.matriz = v if self.matriz is None else np.vstack([self.matriz, v])

    def buscar(self, vector, k=5):
        if self.matriz is None or not len(self.fragmentos):
            return []
        sim = self.matriz @ np.asarray(vector, dtype=np.float32)
        orden = np.argsort(-sim)[:k]
        return [(self.fragmentos[i], float(sim[i])) for i in orden]

    def vaciar(self):
        self.fragmentos, self.matriz = [], None

    def guardar(self, directorio):
        d = Path(directorio)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "vectores.npy", self.matriz)
        (d / "fragmentos.json").write_text(
            json.dumps([f.__dict__ for f in self.fragmentos], ensure_ascii=False), encoding="utf8")

    @classmethod
    def cargar(cls, directorio):
        from .corpus import Fragmento
        d = Path(directorio)
        ind = cls()
        ind.matriz = np.load(d / "vectores.npy")
        ind.fragmentos = [Fragmento(**f) for f in json.loads((d / "fragmentos.json").read_text(encoding="utf8"))]
        return ind


class IndiceChroma(Indice):
    """ChromaDB con índice HNSW y distancia coseno (``pip install chromadb``)."""

    def __init__(self, directorio="indice_chroma", coleccion="corpus"):
        import chromadb  # importación diferida
        self.cliente = chromadb.PersistentClient(path=str(directorio))
        self.nombre = coleccion
        self.col = self.cliente.get_or_create_collection(coleccion, metadata={"hnsw:space": "cosine"})
        self._por_id = {}

    def agregar(self, fragmentos, vectores):
        for f in fragmentos:
            self._por_id[f.id] = f
        self.col.add(ids=[f.id for f in fragmentos], embeddings=np.asarray(vectores).tolist(),
                     documents=[f.texto for f in fragmentos],
                     metadatas=[{"documento": f.documento, "seccion": f.seccion, "version": f.version}
                                for f in fragmentos])

    def buscar(self, vector, k=5):
        from .corpus import Fragmento
        r = self.col.query(query_embeddings=[np.asarray(vector).tolist()], n_results=k)
        out = []
        for i, doc, meta, dist in zip(r["ids"][0], r["documents"][0], r["metadatas"][0], r["distances"][0]):
            f = self._por_id.get(i) or Fragmento(i, doc, meta["documento"], meta["seccion"], meta["version"])
            out.append((f, 1.0 - float(dist)))      # distancia coseno -> similitud
        return out

    def vaciar(self):
        self.cliente.delete_collection(self.nombre)
        self.col = self.cliente.get_or_create_collection(self.nombre, metadata={"hnsw:space": "cosine"})
        self._por_id = {}
