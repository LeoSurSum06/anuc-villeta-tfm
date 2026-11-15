"""Función de recompensa (Ecuación 4.4).

    r_t = clip( alfa * dp / (1 - p_antes) + beta * f_int - gamma * aband , 0, 1 )

con alfa = 0,6, beta = 0,3 y gamma = 0,1. ``dp`` es el cambio en la probabilidad
de dominio de la competencia trabajada, normalizado por el margen disponible
(1 - p_antes); ``f_int`` vale 1 si la actividad se completa y ``aband`` vale 1 si
la persona cierra la sesión durante la actividad (o se corta la conectividad).

Se distinguen dos usos:

* recompensa observable: ``dp`` es la ganancia estimada por BKT. Es la señal con la
  que aprenden las políticas, y la única disponible en el sistema real.
* recompensa de evaluación: ``dp`` se sustituye por la ganancia latente (1 si en ese
  paso la competencia pasa de no dominada a dominada y 0 en otro caso). La calcula el
  simulador y es la métrica primaria de la evaluación, porque ninguna política puede
  mejorarla engañando al estimador.
"""
import numpy as np

from .config import PESOS_RECOMPENSA

_MARGEN_MIN = 1e-3   # evita dividir por cero si p_antes ~ 1


def recompensa(dp, p_antes, f_int, aband, pesos=PESOS_RECOMPENSA):
    """Recompensa de un paso. Acepta escalares o arreglos de NumPy."""
    margen = np.maximum(1.0 - np.asarray(p_antes, dtype=float), _MARGEN_MIN)
    r = (pesos["alfa"] * np.asarray(dp, dtype=float) / margen
         + pesos["beta"] * np.asarray(f_int, dtype=float)
         - pesos["gamma"] * np.asarray(aband, dtype=float))
    return np.clip(r, 0.0, 1.0)


def recompensa_evaluacion(ganancia_latente, p_antes, f_int, aband, pesos=PESOS_RECOMPENSA):
    """Recompensa de evaluación: misma expresión con la ganancia latente."""
    return recompensa(ganancia_latente, p_antes, f_int, aband, pesos)
