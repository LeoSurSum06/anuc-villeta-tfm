"""Espacio de acciones y máscara de seguridad (Subsección 4.2.4).

Ninguna política, por exploratoria que sea, puede elegir una acción fuera del
subconjunto admisible que define la máscara. Las reglas se aplican en este orden:

1. Prioridad obligatoria: mientras p[C7] < 0,6 solo son admisibles acciones de C7.
2. Prerrequisitos: no hay actividades avanzadas de C4 a C6 sin dominar C7 (p >= 0,95).
3. Formato: con preferencia por la voz se excluyen las acciones solo textuales.
4. Pertinencia (suave): se priorizan las competencias de la labor aún no dominadas;
   si no queda ninguna, se mantienen todas las admisibles.
"""
import numpy as np

from .config import ACCIONES, C4_C6, C7, DOMINIO, UMBRAL_C7


def mascara(p, perfil, competencias_labor, umbral_c7=UMBRAL_C7, dominio=DOMINIO):
    """Devuelve la lista de índices de acciones admisibles (Código 4.2)."""
    admisibles = []
    for idx, (k, tipo, formato, dif) in enumerate(ACCIONES):
        if p[C7] < umbral_c7 and k != C7:          # Regla 1: prioridad de C7
            continue
        if k in C4_C6 and dif == "avanzada" and p[C7] < dominio:
            continue                                # Regla 2: prerrequisito
        if perfil["voz"] and formato == "texto_ilustracion":
            continue                                # Regla 3: formato
        admisibles.append(idx)
    # Regla 4: pertinencia suave (nunca deja el conjunto vacío)
    pertinentes = [a for a in admisibles
                   if ACCIONES[a][0] in competencias_labor and p[ACCIONES[a][0]] < dominio]
    return pertinentes if pertinentes else admisibles


# --- Versión indexada: reduce la construcción a O(|A_t|) en lugar de O(336) ---
_COMP = np.array([a[0] for a in ACCIONES])
_AVANZADA = np.array([a[3] == "avanzada" for a in ACCIONES])
_TEXTO = np.array([a[2] == "texto_ilustracion" for a in ACCIONES])
_IDX_POR_COMP = [np.flatnonzero(_COMP == k) for k in range(14)]
_ES_C4_C6 = np.isin(_COMP, C4_C6)


def mascara_np(p, voz, competencias_labor, umbral_c7=UMBRAL_C7, dominio=DOMINIO):
    """Versión con NumPy de ``mascara``; devuelve un arreglo de índices.

    Equivale a la versión de referencia (hay una prueba de equivalencia en
    ``tests/test_acciones.py``) pero evita recorrer en Python las 336 acciones.
    """
    if p[C7] < umbral_c7:
        cand = _IDX_POR_COMP[C7]
    else:
        cand = np.arange(len(ACCIONES))
    ok = np.ones(len(cand), dtype=bool)
    if p[C7] < dominio:
        ok &= ~(_ES_C4_C6[cand] & _AVANZADA[cand])
    if voz:
        ok &= ~_TEXTO[cand]
    adm = cand[ok]
    comp = _COMP[adm]
    pert_mask = np.isin(comp, competencias_labor) & (p[comp] < dominio)
    pert = adm[pert_mask]
    return pert if pert.size else adm


def mascara_lote(P, voz, lab_mask, umbral_c7=UMBRAL_C7, dominio=DOMINIO):
    """Máscara de seguridad para N personas a la vez.

    Parameters
    ----------
    P : (N, 14) probabilidades de dominio estimadas.
    voz : (N,) preferencia de voz (0/1).
    lab_mask : (N, 14) booleano; True en las competencias ligadas a la labor.

    Devuelve una matriz booleana (N, 336) con las acciones admisibles de cada
    persona. Equivale a aplicar ``mascara`` fila a fila.
    """
    c7_bajo = (P[:, C7] < umbral_c7)[:, None]
    base = np.where(c7_bajo, (_COMP == C7)[None, :], True)                  # regla 1
    base &= ~((P[:, C7] < dominio)[:, None] & (_ES_C4_C6 & _AVANZADA)[None, :])  # regla 2
    base &= ~((voz > 0)[:, None] & _TEXTO[None, :])                         # regla 3
    pert = base & lab_mask[:, _COMP] & (P[:, _COMP] < dominio)              # regla 4
    hay = pert.any(axis=1)[:, None]
    return np.where(hay, pert, base)
