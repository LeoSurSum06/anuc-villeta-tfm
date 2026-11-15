"""Las siete políticas de selección comparadas en la evaluación (Tabla 4.11).

Todas operan dentro del subconjunto admisible que define la máscara de seguridad,
de modo que las diferencias entre ellas se deben únicamente al criterio de selección.

Interfaz común
--------------
``seleccionar(ctx)`` recibe un objeto con los campos ``X`` (N, 32), ``P`` (N, 14),
``adm`` (N, 336, booleano), ``voz`` (N,), ``lectura`` (N,) y ``conteo`` (N, 14), y
devuelve un arreglo (N,) con el índice de la acción elegida por cada persona.
``actualizar(ctx, acciones, recompensas)`` procesa las observaciones de la ronda.
Como en el modo fuera de línea, las políticas se actualizan por lotes al final de
cada ronda.
"""
import numpy as np

from .config import (ACCIONES, ALPHA_UCB, D_CONTEXTO, DIFICULTADES, DOMINIO, EPSILON,
                     FORMATOS, LAMBDA_RIDGE, N_ACCIONES, TIPOS, V_THOMPSON)

ACCION_ID = {a: i for i, a in enumerate(ACCIONES)}
_COMP = np.array([a[0] for a in ACCIONES])


def _argmax_aleatorio(valores, rng):
    """argmax con desempate aleatorio (evita sesgo hacia los índices bajos)."""
    mejor = valores.max()
    cand = np.flatnonzero(valores >= mejor - 1e-12)
    return cand[0] if cand.size == 1 else cand[rng.integers(cand.size)]


class Politica:
    nombre = "base"

    def __init__(self, rng):
        self.rng = rng

    def seleccionar(self, ctx):
        raise NotImplementedError

    def actualizar(self, ctx, acciones, recompensas):
        """Por defecto, las políticas sin aprendizaje ignoran las observaciones."""


class Aleatoria(Politica):
    """Elección uniforme entre las acciones admisibles (cota inferior)."""
    nombre = "aleatoria"

    def seleccionar(self, ctx):
        out = np.empty(ctx.X.shape[0], dtype=int)
        for n in range(out.size):
            cand = np.flatnonzero(ctx.adm[n])
            out[n] = cand[self.rng.integers(cand.size)]
        return out


def _elegir_accion(cand_idx, k, tipo, formato, dif):
    """Devuelve la acción pedida si es admisible; si no, la más cercana de la competencia k."""
    cand = set(cand_idx.tolist())
    a = ACCION_ID[(k, tipo, formato, dif)]
    if a in cand:
        return a
    de_k = [i for i in cand_idx if _COMP[i] == k]
    if not de_k:
        return int(cand_idx[0])
    # preferir la misma dificultad y formato; si no, la misma dificultad; si no, la primera
    for i in de_k:
        if ACCIONES[i][3] == dif and ACCIONES[i][1] == tipo:
            return int(i)
    for i in de_k:
        if ACCIONES[i][3] == dif:
            return int(i)
    return int(de_k[0])


class SecuenciaFija(Politica):
    """Orden curricular C1 -> C14 con dificultad creciente (práctica habitual)."""
    nombre = "secuencia_fija"

    def seleccionar(self, ctx):
        out = np.empty(ctx.X.shape[0], dtype=int)
        for n in range(out.size):
            cand = np.flatnonzero(ctx.adm[n])
            comps = np.unique(_COMP[cand])
            sin_dominar = [k for k in comps if ctx.P[n, k] < DOMINIO]
            k = int(sin_dominar[0] if sin_dominar else comps[0])    # primera en el orden
            veces = int(ctx.conteo[n, k])
            dif = DIFICULTADES[0 if veces < 2 else 1 if veces < 4 else 2]  # creciente
            tipo = TIPOS[veces % len(TIPOS)]
            formato = FORMATOS[1] if (ctx.voz[n] or ctx.lectura[n] == 1) else FORMATOS[0]
            out[n] = _elegir_accion(cand, k, tipo, formato, dif)
        return out


class ReglasExpertas(Politica):
    """Elige la competencia con menor p_k y la dificultad según umbrales."""
    nombre = "reglas_expertas"

    def seleccionar(self, ctx):
        out = np.empty(ctx.X.shape[0], dtype=int)
        for n in range(out.size):
            cand = np.flatnonzero(ctx.adm[n])
            comps = np.unique(_COMP[cand])
            k = int(comps[np.argmin(ctx.P[n, comps])])              # menor dominio estimado
            p = ctx.P[n, k]
            dif = DIFICULTADES[0 if p < 0.40 else 1 if p < 0.70 else 2]
            tipo = TIPOS[0 if p < 0.35 else 1 if p < 0.65 else 2 if p < 0.85 else 3]
            formato = FORMATOS[1] if (ctx.voz[n] or ctx.lectura[n] == 1) else FORMATOS[0]
            out[n] = _elegir_accion(cand, k, tipo, formato, dif)
        return out


class EpsilonGreedy(Politica):
    """Bandido no contextual: explotación con exploración epsilon (Sutton y Barto, 2018)."""
    nombre = "epsilon_greedy"

    def __init__(self, rng, epsilon=EPSILON):
        super().__init__(rng)
        self.epsilon = epsilon
        self.suma = np.zeros(N_ACCIONES)
        self.n = np.zeros(N_ACCIONES)

    def seleccionar(self, ctx):
        media = np.where(self.n > 0, self.suma / np.maximum(self.n, 1), 0.0)
        out = np.empty(ctx.X.shape[0], dtype=int)
        for i in range(out.size):
            cand = np.flatnonzero(ctx.adm[i])
            if self.rng.random() < self.epsilon:
                out[i] = cand[self.rng.integers(cand.size)]
            else:
                out[i] = cand[_argmax_aleatorio(media[cand], self.rng)]
        return out

    def actualizar(self, ctx, acciones, recompensas):
        np.add.at(self.suma, acciones, recompensas)
        np.add.at(self.n, acciones, 1)


class Oraculo(Politica):
    """Conoce los parámetros ocultos y elige la acción de mayor recompensa esperada
    inmediata (oráculo miope). Es la cota superior para calcular el regret."""
    nombre = "oraculo"

    def __init__(self, rng, entorno):
        super().__init__(rng)
        self.entorno = entorno

    def seleccionar(self, ctx):
        E = self.entorno.recompensa_esperada_todas()          # (N, 336)
        E = np.where(ctx.adm, E, -np.inf)
        out = np.empty(E.shape[0], dtype=int)
        for n in range(out.size):
            out[n] = _argmax_aleatorio(E[n], self.rng)
        return out


class _Lineal(Politica):
    """Estadísticos suficientes compartidos por LinUCB disjunto y Thompson lineal.

    Para cada acción a se mantiene la inversa A_a^{-1} (A_a = lambda*I + sum x x^T) y
    el vector b_a = sum r*x; los parámetros son theta_a = A_a^{-1} b_a (regresión ridge).
    """

    def __init__(self, rng, lam=LAMBDA_RIDGE, prior_sesgo=0.0):
        super().__init__(rng)
        self.A_inv = np.tile(np.eye(D_CONTEXTO) / lam, (N_ACCIONES, 1, 1))
        self.b = np.zeros((N_ACCIONES, D_CONTEXTO))
        if prior_sesgo:
            # prior informado: recompensa media esperada en el término de sesgo
            self.b[:, -1] = lam * prior_sesgo
        self.theta = np.einsum("ade,ae->ad", self.A_inv, self.b)

    def actualizar(self, ctx, acciones, recompensas):
        X = ctx.X
        for i in range(len(acciones)):
            self._sherman_morrison(int(acciones[i]), X[i], float(recompensas[i]))

    def _sherman_morrison(self, a, x, r):
        Ai = self.A_inv[a]
        Ax = Ai @ x                                           # Ecuación 4.6
        self.A_inv[a] = Ai - np.outer(Ax, Ax) / (1.0 + x @ Ax)
        self.b[a] += r * x
        self.theta[a] = self.A_inv[a] @ self.b[a]

    def _media_var(self, idx, x):
        Ax = self.A_inv[idx] @ x                              # (m, d)
        return self.theta[idx] @ x, Ax @ x


class LinUCB(_Lineal):
    """LinUCB disjunto (Li et al., 2010), Ecuación 4.5."""
    nombre = "linucb"

    def __init__(self, rng, alpha=ALPHA_UCB, lam=LAMBDA_RIDGE, prior_sesgo=0.0):
        super().__init__(rng, lam, prior_sesgo)
        self.alpha = alpha

    def seleccionar(self, ctx):
        out = np.empty(ctx.X.shape[0], dtype=int)
        for n in range(out.size):
            idx = np.flatnonzero(ctx.adm[n])
            media, var = self._media_var(idx, ctx.X[n])
            puntaje = media + self.alpha * np.sqrt(np.maximum(var, 0.0))
            puntaje = puntaje + self.rng.random(idx.size) * 1e-9   # desempate aleatorio
            out[n] = idx[np.argmax(puntaje)]
        return out

    def actualizar(self, ctx, acciones, recompensas):
        """Actualización de Sherman y Morrison (Código 4.3)."""
        super().actualizar(ctx, acciones, recompensas)


class ThompsonLineal(_Lineal):
    """Muestreo de Thompson lineal (Agrawal y Goyal, 2013), Ecuación 4.7.

    En lugar de muestrear theta completo se muestrea directamente la proyección
    x^T theta_tilde, que es una normal univariante con media x^T theta y varianza
    v^2 x^T A^{-1} x; es equivalente para la decisión y mucho más barata.
    """
    nombre = "thompson"

    def __init__(self, rng, v=V_THOMPSON, lam=LAMBDA_RIDGE, prior_sesgo=0.0):
        super().__init__(rng, lam, prior_sesgo)
        self.v = v

    def seleccionar(self, ctx):
        out = np.empty(ctx.X.shape[0], dtype=int)
        for n in range(out.size):
            idx = np.flatnonzero(ctx.adm[n])
            media, var = self._media_var(idx, ctx.X[n])
            muestra = media + self.v * np.sqrt(np.maximum(var, 0.0)) * self.rng.standard_normal(idx.size)
            out[n] = idx[np.argmax(muestra)]
        return out


def crear_politica(nombre, rng, entorno=None, **kw):
    """Fábrica de políticas por nombre."""
    if nombre == "secuencia_fija":
        return SecuenciaFija(rng)
    if nombre == "aleatoria":
        return Aleatoria(rng)
    if nombre == "reglas_expertas":
        return ReglasExpertas(rng)
    if nombre == "epsilon_greedy":
        return EpsilonGreedy(rng, **kw)
    if nombre == "oraculo":
        return Oraculo(rng, entorno)
    if nombre == "linucb":
        return LinUCB(rng, **kw)
    if nombre == "thompson":
        return ThompsonLineal(rng, **kw)
    raise ValueError(f"Política desconocida: {nombre}")
