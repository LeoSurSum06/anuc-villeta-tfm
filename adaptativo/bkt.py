"""Seguimiento del conocimiento bayesiano (BKT), Ecuaciones 4.1 a 4.3.

Cada competencia k tiene una probabilidad de dominio estimada p_k. Tras una
respuesta se calcula la probabilidad posterior (Ecuaciones 4.1 y 4.2) y se aplica
la transición de aprendizaje (Ecuación 4.3). El coste es constante: solo se
actualiza la competencia trabajada.
"""
import numpy as np

from .config import COTA_SG, DOMINIO


def bkt_update(p, correcta, s, g, t):
    """Actualiza la probabilidad de dominio p de una competencia (Código 4.1)."""
    if correcta:   # Ecuación 4.1
        post = p * (1 - s) / (p * (1 - s) + (1 - p) * g)
    else:          # Ecuación 4.2
        post = p * s / (p * s + (1 - p) * (1 - g))
    # Ecuación 4.3: la persona puede aprender durante la actividad
    return post + (1 - post) * t


def bkt_update_vec(p, correcta, s, g, t):
    """Versión vectorizada de ``bkt_update`` para arreglos de NumPy."""
    p = np.asarray(p, dtype=float)
    correcta = np.asarray(correcta, dtype=bool)
    num_c = p * (1 - s)
    den_c = num_c + (1 - p) * g
    num_i = p * s
    den_i = num_i + (1 - p) * (1 - g)
    post = np.where(correcta, num_c / den_c, num_i / den_i)
    return post + (1 - post) * t


def parametros_identificables(s, g):
    """Comprueba la condición de identificabilidad de Baker et al. (2008).

    Se exige s <= 0,30, g <= 0,30 y, por tanto, s + g < 1, de modo que quien
    domina la competencia nunca tiene menor probabilidad de acierto que quien
    no la domina.
    """
    s = np.asarray(s)
    g = np.asarray(g)
    return bool(np.all(s <= COTA_SG + 1e-12) and np.all(g <= COTA_SG + 1e-12)
                and np.all(s + g < 1.0))


def dominada(p, umbral=DOMINIO):
    """Una competencia se considera dominada cuando p >= 0,95."""
    return np.asarray(p) >= umbral
