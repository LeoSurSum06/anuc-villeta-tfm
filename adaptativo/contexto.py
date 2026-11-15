"""Vector de contexto x_t de dimensión d = 32 (Tabla 4.6).

Grupos: estado de conocimiento (14), labor con categoría de referencia (4),
edad con categoría de referencia (3), condiciones autodeclaradas (4),
interacción (2), situación (4) y sesgo (1). Las variables categóricas usan una
categoría de referencia para que la matriz de diseño no sea colineal con el sesgo.
"""
import numpy as np

from .config import CONDICIONES, D_CONTEXTO, EDADES, K, LABORES

_LABOR_IDX = {l: i - 1 for i, l in enumerate(LABORES)}   # la primera es la referencia
_EDAD_IDX = {e: i - 1 for i, e in enumerate(EDADES)}


def construir_contexto(p, perfil, situacion, consentimiento=True):
    """Construye x_t para una persona.

    Parameters
    ----------
    p : arreglo de 14 probabilidades de dominio estimadas.
    perfil : dict con ``labor``, ``edad``, ``lectura`` (1-3), ``voz`` (0/1) y las
        condiciones declaradas (``hipertension``, ``diabetes``, ``osteomuscular``,
        ``embarazo``), que pueden faltar.
    situacion : dict con ``conectividad``, ``franja``, ``dias`` y ``sesiones``,
        todas ya normalizadas en [0, 1].
    consentimiento : si es False, las cuatro condiciones valen cero (la persona
        no consiente el tratamiento de datos sensibles).
    """
    x = np.zeros(D_CONTEXTO)
    x[:K] = p
    o = K
    if perfil["labor"] in _LABOR_IDX and _LABOR_IDX[perfil["labor"]] >= 0:
        x[o + _LABOR_IDX[perfil["labor"]]] = 1.0
    o += len(LABORES) - 1
    if perfil["edad"] in _EDAD_IDX and _EDAD_IDX[perfil["edad"]] >= 0:
        x[o + _EDAD_IDX[perfil["edad"]]] = 1.0
    o += len(EDADES) - 1
    if consentimiento:
        for j, c in enumerate(CONDICIONES):
            x[o + j] = float(perfil.get(c, 0))
    o += len(CONDICIONES)
    x[o] = float(perfil["voz"])
    x[o + 1] = (perfil["lectura"] - 1) / 2.0
    o += 2
    x[o] = situacion["conectividad"]
    x[o + 1] = situacion["franja"]
    x[o + 2] = situacion["dias"]
    x[o + 3] = situacion["sesiones"]
    o += 4
    x[o] = 1.0           # sesgo
    return x


def contexto_lote(P, perfiles_arr, situacion, consentimiento=True):
    """Versión por lotes. ``perfiles_arr`` es el resultado de ``codificar_perfiles``.

    Devuelve una matriz (N, 32). ``situacion`` es un dict de arreglos de longitud N.
    """
    N = P.shape[0]
    X = np.zeros((N, D_CONTEXTO))
    X[:, :K] = P
    o = K
    X[:, o:o + len(LABORES) - 1] = perfiles_arr["labor_oh"]
    o += len(LABORES) - 1
    X[:, o:o + len(EDADES) - 1] = perfiles_arr["edad_oh"]
    o += len(EDADES) - 1
    if consentimiento:
        X[:, o:o + len(CONDICIONES)] = perfiles_arr["condiciones"]
    o += len(CONDICIONES)
    X[:, o] = perfiles_arr["voz"]
    X[:, o + 1] = (perfiles_arr["lectura"] - 1) / 2.0
    o += 2
    X[:, o] = situacion["conectividad"]
    X[:, o + 1] = situacion["franja"]
    X[:, o + 2] = situacion["dias"]
    X[:, o + 3] = situacion["sesiones"]
    X[:, o + 4] = 1.0
    return X


def codificar_perfiles(perfiles):
    """Convierte la lista de perfiles en arreglos para el contexto por lotes."""
    n = len(perfiles)
    labor_oh = np.zeros((n, len(LABORES) - 1))
    edad_oh = np.zeros((n, len(EDADES) - 1))
    cond = np.zeros((n, len(CONDICIONES)))
    voz = np.zeros(n)
    lectura = np.zeros(n)
    for i, pf in enumerate(perfiles):
        j = _LABOR_IDX[pf["labor"]]
        if j >= 0:
            labor_oh[i, j] = 1.0
        j = _EDAD_IDX[pf["edad"]]
        if j >= 0:
            edad_oh[i, j] = 1.0
        for c, nombre in enumerate(CONDICIONES):
            cond[i, c] = float(pf.get(nombre, 0))
        voz[i] = pf["voz"]
        lectura[i] = pf["lectura"]
    return {"labor_oh": labor_oh, "edad_oh": edad_oh, "condiciones": cond,
            "voz": voz, "lectura": lectura}
