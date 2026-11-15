"""Simulador de las personas aprendices (Subsección 4.3.2).

Cada perfil tiene, para cada competencia, un estado latente binario que se inicializa
con probabilidad p0. Cuando la política elige una actividad sobre la competencia k, el
entorno genera la respuesta con probabilidad de acierto 1 - s_k si la competencia está
dominada y g_k si no lo está; a continuación, si no estaba dominada, pasa a estarlo con
probabilidad t_k. Una actividad avanzada sobre una competencia con dominio bajo reduce a
la mitad la probabilidad de aprendizaje y duplica la de abandono (zona de desarrollo
próximo). El sistema solo ve la respuesta y actualiza su estimación con BKT.

Supuestos del simulador que la memoria no fija y que se documentan aquí:

* cada interacción es una sesión; entre sesiones pasan 1, 2 o 3 días (``DIAS_ENTRE_SESIONES``);
* el tipo de actividad y el formato no modifican la dinámica (solo la dificultad lo hace);
* una sesión cortada por falta de conectividad cuenta como abandono y no produce
  respuesta ni aprendizaje;
* el diagnóstico inicial fija la estimación de partida igual a p0 (en E2, sin
  diagnóstico, todas parten de 0,25).

Los resultados aleatorios de cada ronda (respuesta, aprendizaje, abandono, corte, olvido)
se sortean con un flujo que depende solo de (semilla, escenario), de modo que todas las
políticas se enfrentan a los mismos números aleatorios (números aleatorios comunes) y las
diferencias pareadas por semilla son más precisas.
"""
import numpy as np

from .acciones import mascara_lote
from .bkt import bkt_update_vec
from .config import (ACCIONES, BKT_DEFECTO, COMPETENCIAS_LABOR, DIAS_ENTRE_SESIONES,
                     DOMINIO, ESCENARIOS, K, LABORES, N_ACCIONES, N_RONDAS, OLVIDO_DIAS,
                     OLVIDO_PROB_DIA, PESOS_RECOMPENSA)
from .contexto import codificar_perfiles, contexto_lote
from .recompensa import recompensa

_COMP = np.array([a[0] for a in ACCIONES])
_DIF_AV = np.array([a[3] == "avanzada" for a in ACCIONES])
_ESC_ID = {e: i for i, e in enumerate(ESCENARIOS)}


class Contexto:
    """Estado observable de una ronda que reciben las políticas."""
    __slots__ = ("X", "P", "adm", "voz", "lectura", "conteo")


class Entorno:
    def __init__(self, perfiles, escenario="E1", semilla=0, n_rondas=N_RONDAS):
        self.perfiles = perfiles
        self.escenario = escenario
        self.semilla = semilla
        self.n_rondas = n_rondas
        N = self.N = len(perfiles)
        self.p0 = np.array([p["p0"] for p in perfiles])
        self.s = np.array([p["s"] for p in perfiles])
        self.g = np.array([p["g"] for p in perfiles])
        self.t = np.array([p["t"] for p in perfiles])
        self.abandono = np.array([p["abandono"] for p in perfiles])
        self.p_corte = np.array([p["p_corte"] for p in perfiles])
        self.cod = codificar_perfiles(perfiles)
        self.voz = self.cod["voz"]
        self.lectura = self.cod["lectura"]
        self.lab_mask = np.zeros((N, K), dtype=bool)
        for i, p in enumerate(perfiles):
            self.lab_mask[i, COMPETENCIAS_LABOR[p["labor"]]] = True
        # flujo común de números aleatorios (independiente de la política)
        self.rng = np.random.default_rng([semilla, _ESC_ID[escenario], 12345])
        self.z = self.rng.random((N, K)) < self.p0          # estado latente
        self.P = np.full((N, K), 0.25) if escenario == "E2" else self.p0.copy()
        self.dia = np.zeros(N)
        self.ultima = np.zeros((N, K))                       # último día de práctica
        self.conteo = np.zeros((N, K))
        self.sesiones = np.zeros(N)
        self.conectividad = np.full(N, 0.9)                  # estimación móvil de conectividad
        self.ronda = 0
        # métricas
        self.sum_eval = 0.0
        self.sum_obs = 0.0
        self.sum_regret = 0.0
        self.violaciones = 0
        self.ronda_c7 = np.full(N, np.nan)
        self.ronda_80 = np.full(N, np.nan)
        self._u = None
        self._E = None

    # ------------------------------------------------------------------ ronda
    def estado(self):
        """Avanza el calendario, aplica el olvido (E6) y devuelve el contexto de la ronda."""
        N = self.N
        u = self._u = {
            "resp": self.rng.random(N), "aprende": self.rng.random(N),
            "aband": self.rng.random(N), "corte": self.rng.random(N),
            "olvido": self.rng.random((N, K)),
            "franja": self.rng.random(N),
            "salto": self.rng.integers(0, len(DIAS_ENTRE_SESIONES), N),
        }
        salto = np.array(DIAS_ENTRE_SESIONES)[u["salto"]].astype(float)
        self.dia += salto
        if self.escenario == "E6":                           # no estacionariedad
            sin_practica = self.dia[:, None] - self.ultima
            dias_olvido = np.minimum(salto[:, None], np.maximum(sin_practica - OLVIDO_DIAS, 0.0))
            prob = 1.0 - (1.0 - OLVIDO_PROB_DIA) ** dias_olvido
            self.z &= ~(u["olvido"] < prob)
        ctx = Contexto()
        situacion = {"conectividad": self.conectividad, "franja": u["franja"],
                     "dias": np.minimum(salto / 7.0, 1.0),
                     "sesiones": np.minimum(self.sesiones / self.n_rondas, 1.0)}
        ctx.X = contexto_lote(self.P, self.cod, situacion)
        ctx.P = self.P
        ctx.adm = mascara_lote(self.P, self.voz, self.lab_mask)
        ctx.voz = self.voz
        ctx.lectura = self.lectura
        ctx.conteo = self.conteo
        self._E = None
        return ctx

    def recompensa_esperada_todas(self):
        """Recompensa de evaluación esperada de cada acción para cada persona, (N, 336).

        Se calcula con los parámetros ocultos y el estado latente actuales; la usan el
        oráculo y el cálculo del regret. Con completar = 1 - q (q: probabilidad de que la
        actividad se interrumpa) y g la probabilidad de que se produzca aprendizaje:
        E[r*] = (1-q) * (g * r_ganancia + (1-g) * beta) + q * r_abandono.
        """
        if self._E is not None:
            return self._E
        a, b, c = PESOS_RECOMPENSA["alfa"], PESOS_RECOMPENSA["beta"], PESOS_RECOMPENSA["gamma"]
        z, t = self.z, self.t
        N = self.N
        E = np.empty((N, N_ACCIONES))
        for av in (False, True):
            sel = _DIF_AV == av
            comp = _COMP[sel]
            adverso = (~z[:, comp]) if av else np.zeros((N, comp.size), dtype=bool)
            q_u = np.clip(self.abandono[:, None] * (1.0 + adverso), 0.0, 0.95)
            q = 1.0 - (1.0 - q_u) * (1.0 - self.p_corte[:, None])
            t_ef = t[:, comp] * (0.5 if av else 1.0) * (~z[:, comp])
            r_gan = np.clip(a * 1.0 / np.maximum(1.0 - self.P[:, comp], 1e-3) + b, 0.0, 1.0)
            r_aband = np.clip(-c, 0.0, 1.0)
            E[:, sel] = (1.0 - q) * (t_ef * r_gan + (1.0 - t_ef) * b) + q * r_aband
        self._E = E
        return E

    def paso(self, ctx, acciones):
        """Ejecuta las acciones elegidas; devuelve las recompensas observables."""
        u, N = self._u, self.N
        idx = np.arange(N)
        acciones = np.asarray(acciones, dtype=int)
        self.violaciones += int((~ctx.adm[idx, acciones]).sum())
        # regret frente al oráculo miope (mismos parámetros ocultos)
        E = self.recompensa_esperada_todas()
        mejor = np.where(ctx.adm, E, -np.inf).max(axis=1)
        self.sum_regret += float((mejor - E[idx, acciones]).sum())

        k = _COMP[acciones]
        avanzada = _DIF_AV[acciones]
        z_k = self.z[idx, k]
        adverso = avanzada & ~z_k
        q_user = np.clip(self.abandono * (1.0 + adverso), 0.0, 0.95)
        abandono = u["aband"] < q_user
        corte = u["corte"] < self.p_corte
        interrumpida = abandono | corte
        completa = ~interrumpida

        p_ac = np.where(z_k, 1.0 - self.s[idx, k], self.g[idx, k])
        correcta = u["resp"] < p_ac
        t_ef = self.t[idx, k] * np.where(avanzada, 0.5, 1.0)
        aprende = completa & ~z_k & (u["aprende"] < t_ef)
        ganancia = aprende.astype(float)

        # estimación BKT del sistema (solo si hay respuesta)
        p_antes = self.P[idx, k].copy()
        p_nuevo = bkt_update_vec(p_antes, correcta, BKT_DEFECTO["s"], BKT_DEFECTO["g"], BKT_DEFECTO["t"])
        dp = np.where(completa, p_nuevo - p_antes, 0.0)
        self.P[idx, k] = np.where(completa, p_nuevo, p_antes)

        r_obs = recompensa(dp, p_antes, completa.astype(float), interrumpida.astype(float))
        r_eval = recompensa(ganancia, p_antes, completa.astype(float), interrumpida.astype(float))
        self.sum_obs += float(r_obs.sum())
        self.sum_eval += float(r_eval.sum())

        # actualización del estado latente y del calendario
        self.z[idx, k] = z_k | aprende
        self.ultima[idx, k] = self.dia
        self.conteo[idx, k] += 1
        self.sesiones += 1
        self.conectividad = 0.9 * self.conectividad + 0.1 * (~corte)
        self.ronda += 1
        dom = self.z[:, 6]
        self.ronda_c7 = np.where(np.isnan(self.ronda_c7) & dom, self.ronda, self.ronda_c7)
        amplio = self.z.sum(axis=1) >= int(np.ceil(0.8 * K))
        self.ronda_80 = np.where(np.isnan(self.ronda_80) & amplio, self.ronda, self.ronda_80)
        return r_obs

    # --------------------------------------------------------------- métricas
    def metricas(self):
        N, R = self.N, self.n_rondas
        c7 = np.where(np.isnan(self.ronda_c7), R, self.ronda_c7)
        p80 = np.where(np.isnan(self.ronda_80), R, self.ronda_80)
        amplio = self.z.sum(axis=1) >= int(np.ceil(0.8 * K))
        return {
            "recompensa_eval": self.sum_eval / (N * R),
            "recompensa_obs": self.sum_obs / (N * R),
            "regret": self.sum_regret / N,
            "pasos_c7": float(c7.mean()),
            "pasos_80": float(p80.mean()),
            "frac_80": float(amplio.mean()),
            "calibracion": float(np.abs(self.P - self.z).mean()),
            "violaciones": self.violaciones,
        }


def simular(perfiles, escenario, semilla, nombre_politica, n_rondas=N_RONDAS, **kw):
    """Ejecuta una simulación completa y devuelve el diccionario de métricas."""
    from .politicas import crear_politica
    env = Entorno(perfiles, escenario, semilla, n_rondas)
    pol_id = ["secuencia_fija", "aleatoria", "reglas_expertas", "epsilon_greedy",
              "oraculo", "linucb", "thompson"].index(nombre_politica)
    rng = np.random.default_rng([semilla, _ESC_ID[escenario], pol_id, 777])
    pol = crear_politica(nombre_politica, rng, entorno=env, **kw)
    for _ in range(n_rondas):
        ctx = env.estado()
        acc = pol.seleccionar(ctx)
        r_obs = env.paso(ctx, acc)
        pol.actualizar(ctx, acc, r_obs)
    return env.metricas()
