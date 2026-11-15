"""Recomendador de rutas: problema de orientación con ventanas horarias (OPTW).

Formulación (Ecuación 4.8, Vansteenwegen et al., 2011). El nodo 0 es la llegada del
grupo y el último nodo su salida; los nodos intermedios son los puntos de interés (POI),
cada uno con puntuación S_i, duración de visita d_i y ventana horaria [e_i, l_i]. Se
busca la ruta que maximice la puntuación total visitando cada punto como máximo una vez,
respetando las ventanas (se permite esperar) y el tiempo disponible T_max.

Se incluyen dos métodos:

* ``resolver_heuristica``: inserción por cociente puntuación/incremento de tiempo,
  seguida de búsqueda local iterada (2-opt, inserción y sustitución de puntos).
* ``resolver_exacto``: programación dinámica sobre los subconjuntos de puntos visitados;
  para cada subconjunto y cada último punto conserva el instante mínimo de fin (con
  espera permitida, llegar antes nunca empeora la solución, por lo que esa dominancia
  es exacta). Es práctico hasta unos 15 a 18 puntos.

Unidades: tiempos en minutos, coordenadas en kilómetros.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Instancia:
    """Datos de una finca. Los nodos 0 y n+1 son la entrada y la salida."""
    coords: np.ndarray            # (n+2, 2) en km; entrada y salida en la misma posición
    puntuacion: np.ndarray        # (n+2,) interés 1-5 (0 en entrada y salida)
    duracion: np.ndarray          # (n+2,) minutos de visita
    ventana: np.ndarray           # (n+2, 2) [e_i, l_i] en minutos desde la llegada
    t_max: float                  # tiempo disponible del grupo, en minutos
    velocidad_kmh: float = 3.0    # velocidad de caminata
    inseguros: set = field(default_factory=set)   # tramos (i, j) excluidos del grafo

    @property
    def n(self):
        return len(self.puntuacion) - 2

    def tiempos(self):
        """Matriz de tiempos de viaje en minutos. Los tramos inseguros valen infinito."""
        d = np.linalg.norm(self.coords[:, None, :] - self.coords[None, :, :], axis=2)
        t = d / self.velocidad_kmh * 60.0
        for (i, j) in self.inseguros:
            t[i, j] = np.inf
        return t


def generar_instancia(n_poi, semilla):
    """Genera una instancia plausible para una finca (el mismo n y semilla dan lo mismo)."""
    rng = np.random.default_rng(semilla)
    coords = rng.uniform(0.0, 1.5, size=(n_poi, 2))             # finca de 1,5 x 1,5 km
    entrada = np.array([[0.0, 0.0]])
    coords = np.vstack([entrada, coords, entrada])
    puntuacion = np.concatenate([[0], rng.integers(1, 6, n_poi), [0]]).astype(float)
    duracion = np.concatenate([[0], rng.integers(10, 46, n_poi), [0]]).astype(float)
    t_max = float(rng.integers(180, 301))
    abre = rng.uniform(0, 0.5 * t_max, n_poi)
    ancho = rng.uniform(0.4 * t_max, 1.0 * t_max, n_poi)
    ventana = np.vstack([[0, t_max], np.column_stack([abre, abre + ancho]), [0, t_max]])
    return Instancia(coords, puntuacion, duracion, ventana, t_max)


# ---------------------------------------------------------------------------
# Evaluación de una ruta
# ---------------------------------------------------------------------------
def simular_ruta(inst, T, ruta):
    """Devuelve (factible, tiempo_final) de una ruta [0, i1, ..., ik, n+1].

    Se espera, si hace falta, hasta la apertura de cada ventana; la ruta es factible si se
    llega a cada punto antes de que cierre su ventana y la salida ocurre dentro de t_max.
    """
    t = 0.0
    for a, b in zip(ruta[:-1], ruta[1:]):
        t = t + inst.duracion[a] + T[a, b]
        if t > inst.ventana[b, 1] + 1e-9:
            return False, np.inf
        t = max(t, inst.ventana[b, 0])
    return t <= inst.t_max + 1e-9, t


def puntuacion_ruta(inst, ruta):
    return float(inst.puntuacion[ruta].sum())


# ---------------------------------------------------------------------------
# Heurística: inserción + búsqueda local iterada
# ---------------------------------------------------------------------------
def _insercion(inst, T, ruta, candidatos):
    """Añade, mientras quepan, el punto con mayor cociente puntuación / incremento de tiempo."""
    ruta = list(ruta)
    _, t_actual = simular_ruta(inst, T, ruta)
    while True:
        mejor = None
        for c in candidatos:
            if c in ruta:
                continue
            for pos in range(1, len(ruta)):
                nueva = ruta[:pos] + [c] + ruta[pos:]
                ok, t_nuevo = simular_ruta(inst, T, nueva)
                if not ok:
                    continue
                razon = inst.puntuacion[c] / max(t_nuevo - t_actual, 1e-6)
                if mejor is None or razon > mejor[0]:
                    mejor = (razon, nueva, t_nuevo)
        if mejor is None:
            return ruta
        _, ruta, t_actual = mejor


def _dos_opt(inst, T, ruta):
    """Intercambio de dos aristas para acortar la ruta (Croes, 1958)."""
    mejoro = True
    _, t_mejor = simular_ruta(inst, T, ruta)
    while mejoro:
        mejoro = False
        for i in range(1, len(ruta) - 2):
            for j in range(i + 1, len(ruta) - 1):
                nueva = ruta[:i] + ruta[i:j + 1][::-1] + ruta[j + 1:]
                ok, t_nuevo = simular_ruta(inst, T, nueva)
                if ok and t_nuevo < t_mejor - 1e-9:
                    ruta, t_mejor, mejoro = nueva, t_nuevo, True
    return ruta


def _sustitucion(inst, T, ruta):
    """Sustituye un punto visitado por uno no visitado de mayor puntuación, si es factible."""
    visitados = ruta[1:-1]
    for v in visitados:
        for c in range(1, inst.n + 1):
            if c in ruta or inst.puntuacion[c] <= inst.puntuacion[v]:
                continue
            nueva = [c if x == v else x for x in ruta]
            ok, _ = simular_ruta(inst, T, nueva)
            if ok:
                return nueva, True
    return ruta, False


def resolver_heuristica(inst, semilla=0, iteraciones=8):
    """Heurística constructiva con búsqueda local iterada (Vansteenwegen et al., 2009).

    Devuelve (ruta, puntuacion). Con la semilla fija es determinista.
    """
    rng = np.random.default_rng(semilla)
    T = inst.tiempos()
    n = inst.n
    todos = list(range(1, n + 1))
    # Se excluyen los puntos inalcanzables (tramos inseguros o ventanas ya cerradas)
    base = [0, n + 1]
    ruta = _insercion(inst, T, base, todos)

    def mejorar(r):
        while True:
            r = _dos_opt(inst, T, r)
            r = _insercion(inst, T, r, todos)
            r, cambio = _sustitucion(inst, T, r)
            if not cambio:
                return r

    ruta = mejorar(ruta)
    mejor, mejor_p = ruta, puntuacion_ruta(inst, ruta)
    for _ in range(iteraciones):                      # perturbación: retirar 1-2 puntos
        interior = mejor[1:-1]
        if not interior:
            break
        k = min(len(interior), int(rng.integers(1, 3)))
        quitar = set(rng.choice(interior, size=k, replace=False).tolist())
        r = [x for x in mejor if x not in quitar]
        r = mejorar(r)
        p = puntuacion_ruta(inst, r)
        if p > mejor_p + 1e-9:
            mejor, mejor_p = r, p
    return mejor, mejor_p


# ---------------------------------------------------------------------------
# Método exacto: programación dinámica sobre subconjuntos
# ---------------------------------------------------------------------------
def resolver_exacto(inst):
    """Óptimo del OPTW por programación dinámica. Devuelve (ruta, puntuacion)."""
    T = inst.tiempos()
    n = inst.n
    fin = n + 1
    S, d, e, l = inst.puntuacion, inst.duracion, inst.ventana[:, 0], inst.ventana[:, 1]
    # estados[(mascara, ultimo)] = (instante_de_fin_de_visita_minimo, anterior)
    mejor_pt, mejor_ruta = 0.0, [0, fin]
    capa = {(0, 0): (0.0, None)}
    padres = {}
    puntos = {(0, 0): 0.0}
    for _ in range(n):
        nueva = {}
        for (m, u), (t_u, _) in capa.items():
            for j in range(1, n + 1):
                if m >> (j - 1) & 1:
                    continue
                llegada = t_u + d[u] + T[u, j]
                if llegada > l[j] + 1e-9:
                    continue
                inicio = max(llegada, e[j])
                # debe poder regresar a la salida dentro de t_max y antes de que cierre
                salida = inicio + d[j] + T[j, fin]
                if salida > inst.t_max + 1e-9 or salida > l[fin] + 1e-9:
                    continue
                m2 = m | (1 << (j - 1))
                clave = (m2, j)
                if clave not in nueva or inicio < nueva[clave][0] - 1e-12:
                    nueva[clave] = (inicio, u)
                    padres[clave] = (m, u)
                    puntos[clave] = puntos[(m, u)] + S[j]
        if not nueva:
            break
        for clave, (t_j, _) in nueva.items():
            if puntos[clave] > mejor_pt + 1e-9:
                mejor_pt = puntos[clave]
                ruta, c = [], clave
                while c[1] != 0:
                    ruta.append(c[1])
                    c = padres[c]
                mejor_ruta = [0] + ruta[::-1] + [fin]
        capa = nueva
    return mejor_ruta, mejor_pt
