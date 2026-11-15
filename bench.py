"""Coste por decisión del motor adaptativo (Subsección 4.4.2).

Cronometra una decisión completa para una persona: construcción del contexto (d = 32), máscara de
seguridad, selección con LinUCB disjunto sobre las acciones admisibles, actualización de Sherman y
Morrison, actualización BKT y cálculo de la recompensa. Se ejecuta en un único hilo.

Por defecto se usa un estado con 88 acciones admisibles (labor Caña/panela, sin preferencia de voz,
con C7 entre 0,6 y 0,95: la regla de prerrequisito retira las 8 acciones avanzadas de C6).

    python bench.py                  # 20 000 decisiones
    python bench.py --n 100000 --alpha 1.0
"""
import argparse
import platform
import time

import numpy as np

from adaptativo import config
from adaptativo.acciones import mascara_np
from adaptativo.bkt import bkt_update
from adaptativo.contexto import construir_contexto
from adaptativo.politicas import LinUCB
from adaptativo.recompensa import recompensa


class _Ctx:
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=20_000)
    ap.add_argument("--alpha", type=float, default=config.ALPHA_UCB)
    ap.add_argument("--semilla", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.semilla)
    pol = LinUCB(rng, alpha=a.alpha)
    perfil = {"labor": "Caña/panela", "edad": "30-45", "lectura": 2, "voz": 0, "hipertension": 1}
    labor = config.COMPETENCIAS_LABOR[perfil["labor"]]
    p = np.full(config.K, 0.30)
    p[config.C7] = 0.80
    situacion = {"conectividad": 0.9, "franja": 0.5, "dias": 0.1, "sesiones": 0.2}
    s, g, t = (config.BKT_DEFECTO[k] for k in ("s", "g", "t"))

    def decision():
        x = construir_contexto(p, perfil, situacion)
        adm = mascara_np(p, perfil["voz"], labor)
        ctx = _Ctx()
        ctx.X, ctx.adm = x[None, :], np.zeros((1, config.N_ACCIONES), dtype=bool)
        ctx.adm[0, adm] = True
        accion = int(pol.seleccionar(ctx)[0])
        k = config.ACCIONES[accion][0]
        correcta = rng.random() < 0.6
        p_antes = p[k]
        p[k] = bkt_update(p_antes, correcta, s, g, t)
        r = float(recompensa(p[k] - p_antes, p_antes, 1.0, 0.0))
        pol._sherman_morrison(accion, x, r)
        p[k] = min(p[k], 0.9)       # mantiene el estado de prueba estacionario
        return adm.size

    for _ in range(500):            # calentamiento
        decision()
    tiempos = np.empty(a.n)
    for i in range(a.n):
        t0 = time.perf_counter_ns()
        n_adm = decision()
        tiempos[i] = (time.perf_counter_ns() - t0) / 1e6
    print(f"Acciones admisibles: {n_adm}")
    print(f"{a.n} decisiones | media {tiempos.mean():.3f} ms | p50 {np.percentile(tiempos, 50):.3f} ms | "
          f"p95 {np.percentile(tiempos, 95):.3f} ms | p99 {np.percentile(tiempos, 99):.3f} ms")
    print(f"Equipo: {platform.processor() or platform.machine()} | Python {platform.python_version()} "
          f"| NumPy {np.__version__}")


if __name__ == "__main__":
    main()
