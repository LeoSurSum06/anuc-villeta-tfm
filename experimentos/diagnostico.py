"""Diagnóstico de la conducta de cada política en el escenario base (Subsección 4.4.6).

Mide, por política, qué proporción de las actividades elegidas son (a) avanzadas sobre una
competencia aún no dominada (penalizadas por el simulador con la mitad de aprendizaje y el doble
de abandono), (b) sobre una competencia que la persona ya domina (práctica sin aprendizaje
posible) y (c) completadas. Ayuda a explicar las diferencias de la Tabla 4.18.

    python -m experimentos.diagnostico --semillas 0-4
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from adaptativo import config
from adaptativo.entorno import Entorno, _COMP, _DIF_AV, _ESC_ID
from adaptativo.perfiles import generar_perfiles
from adaptativo.politicas import crear_politica

from .ejecutar import parsear_semillas


def diagnosticar(semillas, escenario="E1"):
    filas = []
    for nombre in config.POLITICAS:
        adv, dominada, completa = [], [], []
        for s in semillas:
            env = Entorno(generar_perfiles(escenario=escenario, semilla=s), escenario, s)
            pid = config.POLITICAS.index(nombre)
            pol = crear_politica(nombre, np.random.default_rng([s, _ESC_ID[escenario], pid, 777]), entorno=env)
            for _ in range(config.N_RONDAS):
                ctx = env.estado()
                acc = pol.seleccionar(ctx)
                k = _COMP[acc]
                z = env.z[np.arange(env.N), k]
                adverso = _DIF_AV[acc] & ~z
                u = env._u
                q = np.clip(env.abandono * (1.0 + adverso), 0.0, 0.95)
                interrumpida = (u["aband"] < q) | (u["corte"] < env.p_corte)
                adv.append(adverso.mean())
                dominada.append(z.mean())
                completa.append((~interrumpida).mean())
                pol.actualizar(ctx, acc, env.paso(ctx, acc))
        filas.append({"politica": nombre, "avanzada_sobre_no_dominada": np.mean(adv),
                      "sobre_competencia_ya_dominada": np.mean(dominada), "completadas": np.mean(completa)})
    return pd.DataFrame(filas)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semillas", default="0-4")
    ap.add_argument("--escenario", default="E1")
    ap.add_argument("--salida", default="experimentos/registros/diagnostico.csv")
    a = ap.parse_args()
    t = diagnosticar(parsear_semillas(a.semillas), a.escenario)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    t.to_csv(a.salida, index=False)
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
