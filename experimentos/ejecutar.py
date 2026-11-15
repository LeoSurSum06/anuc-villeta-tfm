"""Ejecuta el experimento de simulación (Subsección 4.4.1).

Por defecto: 6 escenarios x 7 políticas x 50 semillas (0 a 49) = 2100 ejecuciones; cada
una simula 100 perfiles durante 200 rondas (20 000 decisiones), es decir, 42 000 000
decisiones en total. Los resultados se añaden a un CSV y la ejecución puede reanudarse:
las combinaciones ya registradas se omiten.

Ejemplos
--------
    python -m experimentos.ejecutar                              # experimento completo
    python -m experimentos.ejecutar --workers 2                  # con dos procesos
    python -m experimentos.ejecutar --semillas 0-9 --escenarios E1   # ejecución piloto
    python -m experimentos.ejecutar --semillas 0-2 --rondas 50       # prueba rápida
    python -m experimentos.ejecutar --sensibilidad                   # Tabla 4.19
"""
import argparse
import csv
import json
import multiprocessing as mp
import subprocess
import time
from pathlib import Path

from adaptativo import config
from adaptativo.entorno import simular
from adaptativo.perfiles import generar_perfiles, huella, validar_poblacion

CAMPOS = ["escenario", "semilla", "politica", "parametro", "valor_parametro", "n_rondas",
          "recompensa_eval", "recompensa_obs", "regret", "pasos_c7", "pasos_80", "frac_80",
          "calibracion", "violaciones", "huella_poblacion", "version_codigo",
          "version_config", "segundos"]

SENSIBILIDAD = [("linucb", "alpha", v) for v in (0.1, 0.3, 1.0)] + \
               [("thompson", "v", v) for v in (0.1, 0.3, 1.0)] + [("secuencia_fija", "", "")]


def version_codigo():
    """Hash de la versión del código, para la trazabilidad de cada ejecución."""
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "sin-git"


def parsear_semillas(txt):
    """'0-49' -> 0..49; '0,3,7' -> [0, 3, 7]."""
    if "-" in txt:
        a, b = txt.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in txt.split(",")]


def _tarea(args):
    escenario, semilla, trabajos, n_rondas, ver = args
    perfiles = generar_perfiles(escenario=escenario, semilla=semilla)
    comprob = validar_poblacion(perfiles)
    assert all(comprob.values()), f"población inválida: {comprob}"
    h = huella(perfiles)
    filas = []
    for politica, parametro, valor in trabajos:
        kw = {parametro: valor} if parametro else {}
        t0 = time.perf_counter()
        m = simular(perfiles, escenario, semilla, politica, n_rondas=n_rondas, **kw)
        filas.append({"escenario": escenario, "semilla": semilla, "politica": politica,
                      "parametro": parametro, "valor_parametro": valor, "n_rondas": n_rondas,
                      **{k: round(v, 6) if isinstance(v, float) else v for k, v in m.items()},
                      "huella_poblacion": h, "version_codigo": ver,
                      "version_config": config.VERSION_CONFIG,
                      "segundos": round(time.perf_counter() - t0, 3)})
    return filas


def _leer_hechos(ruta):
    hechos = set()
    if Path(ruta).exists():
        with open(ruta, encoding="utf8") as f:
            for r in csv.DictReader(f):
                hechos.add((r["escenario"], int(r["semilla"]), r["politica"], r["parametro"],
                            str(r["valor_parametro"])))
    return hechos


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--semillas", default="0-49")
    ap.add_argument("--escenarios", default=",".join(config.ESCENARIOS))
    ap.add_argument("--politicas", default=",".join(config.POLITICAS))
    ap.add_argument("--rondas", type=int, default=config.N_RONDAS)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--salida", default="experimentos/registros/resultados.csv")
    ap.add_argument("--sensibilidad", action="store_true",
                    help="análisis de sensibilidad de alpha_ucb y v (escenario base, semillas 0-9)")
    a = ap.parse_args()

    if a.sensibilidad:
        semillas, escenarios, trabajos = list(range(10)), ["E1"], SENSIBILIDAD
        salida = a.salida.replace("resultados.csv", "sensibilidad.csv")
    else:
        semillas, escenarios = parsear_semillas(a.semillas), a.escenarios.split(",")
        trabajos = [(p, "", "") for p in a.politicas.split(",")]
        salida = a.salida
    Path(salida).parent.mkdir(parents=True, exist_ok=True)
    hechos = _leer_hechos(salida)
    ver = version_codigo()

    tareas = []
    for esc in escenarios:
        for s in semillas:
            pend = [t for t in trabajos if (esc, s, t[0], t[1], str(t[2])) not in hechos]
            if pend:
                tareas.append((esc, s, pend, a.rondas, ver))
    total = sum(len(t[2]) for t in tareas)
    print(f"{total} ejecuciones pendientes ({len(hechos)} ya registradas) -> {salida}")
    if not tareas:
        return
    nuevo = not Path(salida).exists()
    t0 = time.time()
    hechas = 0
    with open(salida, "a", newline="", encoding="utf8") as f:
        w = csv.DictWriter(f, fieldnames=CAMPOS)
        if nuevo:
            w.writeheader()
        iterador = (mp.Pool(a.workers).imap_unordered(_tarea, tareas) if a.workers > 1
                    else map(_tarea, tareas))
        for filas in iterador:
            w.writerows(filas)
            f.flush()
            hechas += len(filas)
            print(f"  {hechas}/{total} ({time.time() - t0:.0f} s)", end="\r")
    print(f"\nTerminado: {hechas} ejecuciones en {time.time() - t0:.0f} s")
    cfg = {"version_config": config.VERSION_CONFIG, "version_codigo": ver,
           "pesos_recompensa": config.PESOS_RECOMPENSA, "bkt": config.BKT_DEFECTO,
           "alpha_ucb": config.ALPHA_UCB, "v_thompson": config.V_THOMPSON,
           "lambda": config.LAMBDA_RIDGE, "epsilon": config.EPSILON,
           "n_perfiles": config.N_PERFILES, "n_rondas": a.rondas}
    Path(salida).with_name("config.json").write_text(json.dumps(cfg, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
