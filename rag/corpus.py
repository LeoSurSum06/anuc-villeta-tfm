"""Gobernanza y fragmentación del corpus documental (Subsección 4.2.10).

Cada documento es un archivo ``.md`` o ``.txt`` con una cabecera de metadatos entre líneas
``---`` (ficha de control de la Tabla 4.7). Solo se indexan los documentos con estado
``aprobado``. La huella SHA-256 de cada archivo y la versión del índice permiten reproducir
cualquier respuesta pasada con el índice vigente en su momento.

Ejemplo de cabecera::

    ---
    id: IFRC-2025-01
    titulo: Directrices internacionales de primeros auxilios
    entidad: IFRC
    edicion: 2025
    vigencia: 2026-06-30
    licencia: Derechos reservados; permiso escrito
    estado: aprobado
    revisor: Nombre, profesional de salud, 2026-10-01
    ---
    # Quemaduras
    Texto del documento...
"""
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

CAMPOS_FICHA = ("id", "titulo", "entidad", "edicion", "vigencia", "licencia", "estado", "revisor")
ESTADOS = ("borrador", "en revision", "aprobado", "retirado")


@dataclass
class Documento:
    meta: dict
    texto: str
    huella: str
    ruta: str = ""

    @property
    def aprobado(self):
        return self.meta.get("estado", "").strip().lower() == "aprobado"


@dataclass
class Fragmento:
    id: str
    texto: str
    documento: str
    seccion: str
    version: str
    meta: dict = field(default_factory=dict)


def leer_documento(ruta):
    """Lee un documento con cabecera de metadatos y calcula su huella."""
    bruto = Path(ruta).read_bytes()
    texto = bruto.decode("utf8")
    meta = {}
    m = re.match(r"^---\s*\n(.*?)\n---\s*\n", texto, flags=re.S)
    if m:
        for linea in m.group(1).splitlines():
            if ":" in linea:
                k, v = linea.split(":", 1)
                meta[k.strip().lower()] = v.strip()
        texto = texto[m.end():]
    meta.setdefault("id", Path(ruta).stem)
    meta.setdefault("estado", "borrador")
    return Documento(meta, texto, hashlib.sha256(bruto).hexdigest(), str(ruta))


def cargar_corpus(directorio):
    """Devuelve todos los documentos del directorio (aprobados o no)."""
    rutas = sorted(p for p in Path(directorio).rglob("*") if p.suffix.lower() in (".md", ".txt"))
    return [leer_documento(p) for p in rutas]


def version_indice(documentos):
    """Versión del índice: huella de las huellas de los documentos aprobados."""
    h = hashlib.sha256()
    for d in sorted((d for d in documentos if d.aprobado), key=lambda d: d.meta["id"]):
        h.update(f"{d.meta['id']}:{d.huella}".encode())
    return h.hexdigest()[:12]


def _secciones(texto):
    """Divide por encabezados markdown; devuelve [(sección, texto)]."""
    partes, actual, buf = [], "Introducción", []
    for linea in texto.splitlines():
        m = re.match(r"^#{1,6}\s+(.*)", linea)
        if m:
            if "".join(buf).strip():
                partes.append((actual, "\n".join(buf)))
            actual, buf = m.group(1).strip(), []
        else:
            buf.append(linea)
    if "".join(buf).strip():
        partes.append((actual, "\n".join(buf)))
    return partes


def fragmentar(doc, version, palabras_min=300, palabras_max=500, solapamiento=50):
    """Divide un documento en fragmentos de ``palabras_min`` a ``palabras_max`` palabras.

    Se respetan las secciones; el último fragmento de una sección puede ser más corto. Los
    fragmentos consecutivos de una misma sección se solapan en ``solapamiento`` palabras.
    """
    frags = []
    paso = max(palabras_max - solapamiento, 1)
    for seccion, cuerpo in _secciones(doc.texto):
        palabras = cuerpo.split()
        ini, n = 0, 0
        while ini < len(palabras):
            fin = min(ini + palabras_max, len(palabras))
            trozo = palabras[ini:fin]
            if len(trozo) < palabras_min and frags and ini > 0 and frags[-1].seccion == seccion:
                # cola corta: ya está cubierta por el solapamiento del fragmento anterior
                break
            frags.append(Fragmento(f"{doc.meta['id']}#{seccion[:30]}#{n}", " ".join(trozo),
                                   doc.meta["id"], seccion, version, dict(doc.meta)))
            n += 1
            if fin >= len(palabras):
                break
            ini += paso
    return frags


def fragmentos_aprobados(documentos, **kw):
    """Fragmentos de todos los documentos aprobados, con la versión del índice."""
    version = version_indice(documentos)
    out = []
    for d in documentos:
        if d.aprobado:
            out.extend(fragmentar(d, version, **kw))
    return out, version
