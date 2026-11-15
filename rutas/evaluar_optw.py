"""Comparación de la heurística de rutas con el óptimo exacto (Subsección 4.4.4).

Genera 30 instancias de entre 8 y 15 puntos de interés, resuelve cada una con la heurística
y con la programación dinámica exacta, y registra la brecha relativa entre ambas
puntuaciones, la proporción de instancias en que la heurística alcanza el óptimo y el
tiempo de cálculo en el percentil 95. Criterio de aceptación de la memoria: brecha media
no superior al 5 % y tiempo de la heurística inferior a 100 ms.

Uso:
    python -m rutas.evaluar_optw --instancias 30 --salida experimentos/registros/optw.csv
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np

from .optw import generar_instancia, puntuacion_ruta, resolver_exacto, resolver_heuristica, simular_ruta


def evaluar(n_instancias=30, semilla_base=0, n_min=8, n_max=15):
    filas = []
    rango = n_max - n_min + 1
    for i in range(n_instancias):
        n_poi = n_min + (i % rango)                    # tamaños 8 a 15, de forma cíclica
        inst = generar_instancia(n_poi, semilla_base + i)
        t0 = time.perf_counter()
        ruta_h, p_h = resolver_heuristica(inst, semilla=semilla_base + i)
        t_h = (time.perf_counter() - t0) * 1000.0
        t0 = time.perf_counter()
        ruta_e, p_e = resolver_exacto(inst)
        t_e = (time.perf_counter() - t0) * 1000.0
        T = inst.tiempos()
        assert simular_ruta(inst, T, ruta_h)[0], "la ruta heurística debe ser factible"
        assert p_h <= p_e + 1e-9, "la heurística no puede superar al óptimo"
        brecha = 0.0 if p_e == 0 else (p_e - p_h) / p_e
        filas.append({"instancia": i, "n_poi": n_poi, "puntuacion_exacta": p_e,
                      "puntuacion_heuristica": p_h, "brecha": brecha,
                      "optimo": int(brecha < 1e-12), "ms_heuristica": t_h, "ms_exacto": t_e})
    return filas


def resumen(filas):
    b = np.array([f["brecha"] for f in filas])
    t = np.array([f["ms_heuristica"] for f in filas])
    return {"instancias": len(filas), "brecha_media": float(b.mean()), "brecha_max": float(b.max()),
            "optimos": int(sum(f["optimo"] for f in filas)),
            "ms_heuristica_p95": float(np.percentile(t, 95)),
            "cumple_brecha": bool(b.mean() <= 0.05),
            "cumple_tiempo": bool(np.percentile(t, 95) < 100.0)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instancias", type=int, default=30)
    ap.add_argument("--semilla-base", type=int, default=0)
    ap.add_argument("--salida", default="experimentos/registros/optw.csv")
    a = ap.parse_args()
    filas = evaluar(a.instancias, a.semilla_base)
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    with open(a.salida, "w", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=list(filas[0]))
        w.writeheader()
        w.writerows(filas)
    r = resumen(filas)
    print(f"Instancias: {r['instancias']} | óptimo alcanzado en {r['optimos']} | "
          f"brecha media {100 * r['brecha_media']:.1f} % (máx. {100 * r['brecha_max']:.1f} %) | "
          f"heurística p95 {r['ms_heuristica_p95']:.1f} ms")
    print(f"Cumple brecha <= 5 %: {r['cumple_brecha']} | cumple tiempo < 100 ms: {r['cumple_tiempo']}")
    print(f"Registros en {a.salida}")


if __name__ == "__main__":
    main()
