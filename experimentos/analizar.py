"""Análisis estadístico de los resultados de la simulación (Subsección 4.4.1).

Lee ``resultados.csv`` y produce:

* estadísticos descriptivos por política y escenario (media, desviación típica, media
  intercuartílica IQM e intervalo de confianza al 95 % por bootstrap con remuestreo de
  semillas, 10 000 remuestreos);
* contrastes de Wilcoxon de rangos con signo sobre las diferencias pareadas por semilla
  entre cada política propuesta (LinUCB y Thompson lineal) y cada referencia principal
  (secuencia fija, reglas expertas y epsilon-greedy), con corrección de Holm dentro de
  cada escenario (y, como comprobación de robustez, sobre las 36 comparaciones) y delta
  de Cliff como tamaño del efecto;
* el criterio de la hipótesis H1;
* las tablas equivalentes a 4.17 y 4.18, la figura de diferencias pareadas (Figura 4.8) y,
  con ``--validar-generador``, las comprobaciones de validez del generador (Tabla 4.16).

Uso:
    python -m experimentos.analizar --entrada experimentos/registros/resultados.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROPUESTAS = ["linucb", "thompson"]
REFERENCIAS = ["secuencia_fija", "reglas_expertas", "epsilon_greedy"]
ORDEN = ["oraculo", "reglas_expertas", "secuencia_fija", "aleatoria", "thompson", "linucb", "epsilon_greedy"]
N_BOOT = 10_000


# ---------------------------------------------------------------- utilidades
def iqm(x):
    """Media intercuartílica: media del 50 % central (Agarwal et al., 2021)."""
    return float(stats.trim_mean(np.asarray(x), 0.25))


def bootstrap_ic(x, estadistico=np.mean, n=N_BOOT, rng=None, nivel=0.95):
    """IC por bootstrap percentil remuestreando las semillas."""
    rng = rng or np.random.default_rng(0)
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    if estadistico is np.mean:
        est = x[idx].mean(axis=1)
    else:
        est = np.array([estadistico(x[i]) for i in idx])
    a = (1 - nivel) / 2
    return float(np.quantile(est, a)), float(np.quantile(est, 1 - a))


def delta_cliff(x, y):
    """Delta de Cliff: P(x > y) - P(x < y) entre dos muestras."""
    x, y = np.asarray(x), np.asarray(y)
    mayor = (x[:, None] > y[None, :]).sum()
    menor = (x[:, None] < y[None, :]).sum()
    return float((mayor - menor) / (len(x) * len(y)))


def holm(pvalores):
    """Corrección de Holm-Bonferroni; devuelve los p ajustados en el orden original."""
    p = np.asarray(pvalores, dtype=float)
    orden = np.argsort(p)
    m = len(p)
    aj = np.empty(m)
    maximo = 0.0
    for rango, i in enumerate(orden):
        maximo = max(maximo, (m - rango) * p[i])
        aj[i] = min(1.0, maximo)
    return aj


def wilcoxon_pareado(a, b):
    d = np.asarray(a) - np.asarray(b)
    if np.allclose(d, 0):
        return 1.0
    return float(stats.wilcoxon(a, b, zero_method="wilcox").pvalue)


# ------------------------------------------------------------------ análisis
def descriptivos(df, metrica="recompensa_eval"):
    filas = []
    for (esc, pol), g in df.groupby(["escenario", "politica"]):
        x = g.sort_values("semilla")[metrica].to_numpy()
        lo, hi = bootstrap_ic(x)
        filas.append({"escenario": esc, "politica": pol, "n": len(x), "media": x.mean(),
                      "desv_tipica": x.std(ddof=1) if len(x) > 1 else np.nan,
                      "iqm": iqm(x), "ic95_inf": lo, "ic95_sup": hi})
    return pd.DataFrame(filas)


def contrastes(df, metrica="recompensa_eval"):
    """Seis comparaciones por escenario; Holm por escenario y sobre las 36."""
    filas = []
    for esc, ge in df.groupby("escenario"):
        piv = ge.pivot(index="semilla", columns="politica", values=metrica)
        for prop in PROPUESTAS:
            for ref in REFERENCIAS:
                if prop not in piv or ref not in piv:
                    continue
                x, y = piv[prop].to_numpy(), piv[ref].to_numpy()
                filas.append({"escenario": esc, "propuesta": prop, "referencia": ref,
                              "dif_media": float((x - y).mean()), "desv_dif": float((x - y).std(ddof=1)),
                              "p_wilcoxon": wilcoxon_pareado(x, y), "cliff": delta_cliff(x, y)})
    c = pd.DataFrame(filas)
    if c.empty:
        return c
    c["p_holm"] = c.groupby("escenario")["p_wilcoxon"].transform(lambda s: holm(s.to_numpy()))
    c["p_holm_36"] = holm(c["p_wilcoxon"].to_numpy())
    return c


def evaluar_h1(df, desc, contr, metrica="recompensa_eval"):
    """H1 se respalda si al menos una propuesta supera a la secuencia fija con diferencia
    significativa tras Holm e IC 95 % no solapados, en el escenario base y en al menos tres
    de los cinco escenarios de estrés."""
    ok = {}
    for esc in sorted(df["escenario"].unique()):
        cumple = False
        for prop in PROPUESTAS:
            fila = contr[(contr.escenario == esc) & (contr.propuesta == prop) & (contr.referencia == "secuencia_fija")]
            d_p = desc[(desc.escenario == esc) & (desc.politica == prop)]
            d_f = desc[(desc.escenario == esc) & (desc.politica == "secuencia_fija")]
            if fila.empty or d_p.empty or d_f.empty:
                continue
            f = fila.iloc[0]
            if f.dif_media > 0 and f.p_holm < 0.05 and d_p.iloc[0].ic95_inf > d_f.iloc[0].ic95_sup:
                cumple = True
        ok[esc] = cumple
    estres = [e for e in ok if e != "E1"]
    respaldada = ok.get("E1", False) and sum(ok[e] for e in estres) >= 3
    return respaldada, ok


def tabla_417(desc, contr):
    """Mejor política propuesta frente a la secuencia fija, por escenario (Tabla 4.17)."""
    filas = []
    for esc in sorted(desc["escenario"].unique()):
        d = desc[(desc.escenario == esc) & desc.politica.isin(PROPUESTAS)]
        if d.empty:
            continue
        mejor = d.sort_values("media", ascending=False).iloc[0]
        ref = desc[(desc.escenario == esc) & (desc.politica == "secuencia_fija")].iloc[0]
        c = contr[(contr.escenario == esc) & (contr.propuesta == mejor.politica)
                  & (contr.referencia == "secuencia_fija")].iloc[0]
        filas.append({"Escenario": esc, "Mejor propuesta": mejor.politica,
                      "Propuesta (IC 95 %)": f"{mejor.media:.4f} [{mejor.ic95_inf:.4f}; {mejor.ic95_sup:.4f}]",
                      "Secuencia fija (IC 95 %)": f"{ref.media:.4f} [{ref.ic95_inf:.4f}; {ref.ic95_sup:.4f}]",
                      "Diferencia pareada": f"{c.dif_media:+.4f}", "p (Holm)": f"{c.p_holm:.3g}",
                      "delta de Cliff": f"{c.cliff:+.2f}"})
    return pd.DataFrame(filas)


def tabla_418(df, escenario="E1"):
    """Métricas secundarias por política en el escenario base (Tabla 4.18)."""
    g = df[df.escenario == escenario]
    filas = []
    for pol in ORDEN:
        x = g[g.politica == pol]
        if x.empty:
            continue
        filas.append({"Política": pol, "Recompensa de evaluación (IQM)": round(iqm(x.recompensa_eval), 4),
                      "Recompensa observable": round(x.recompensa_obs.mean(), 4),
                      "Regret": round(x.regret.mean(), 1), "Pasos hasta dominar C7": round(x.pasos_c7.mean(), 1),
                      "Perfiles con ≥ 80 % dominado": f"{100 * x.frac_80.mean():.0f} %",
                      "Error de calibración": round(x.calibracion.mean(), 3)})
    return pd.DataFrame(filas)


def figura_diferencias(df, ruta):
    """Figura 4.8: diferencia pareada con la secuencia fija, con IC 95 % por bootstrap."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    escs = sorted(df["escenario"].unique())
    pols = [p for p in ORDEN if p not in ("secuencia_fija",)]
    fig, axs = plt.subplots(1, len(escs), figsize=(2.6 * len(escs), 3.8), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, esc in zip(axs, escs):
        piv = df[df.escenario == esc].pivot(index="semilla", columns="politica", values="recompensa_eval")
        for i, p in enumerate(pols):
            if p not in piv:
                continue
            d = (piv[p] - piv["secuencia_fija"]).to_numpy()
            lo, hi = bootstrap_ic(d, n=2000)
            ax.errorbar(d.mean(), i, xerr=[[d.mean() - lo], [hi - d.mean()]], fmt="o", capsize=3)
        ax.axvline(0, color="gray", lw=0.8, ls="--")
        ax.set_title(esc)
        ax.set_yticks(range(len(pols)))
        ax.set_yticklabels(pols if esc == escs[0] else [])
        ax.set_xlabel("Dif. pareada")
    fig.tight_layout()
    Path(ruta).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=200)
    plt.close(fig)


# ------------------------------------------- validez del generador (Tabla 4.16)
def validar_generador(semillas=range(50), escenarios=("E1", "E2", "E3", "E4", "E5", "E6")):
    from adaptativo.config import EDADES
    from adaptativo.perfiles import (P_HIPERTENSION, a_dataframe, generar_perfiles, huella,
                                     validar_poblacion)
    n = bal = ident = rep = 0
    desv = {"p0": [], "s": [], "g": [], "t": [], "abandono": [], "lectura1": [], "voz": [], "hipertension": []}
    for esc in escenarios:
        acum = {k: [] for k in desv}
        for s in semillas:
            pf = generar_perfiles(escenario=esc, semilla=s)
            v = validar_poblacion(pf)
            n += 1
            bal += v["balance_estratos"]
            ident += v["identificabilidad"]
            rep += huella(pf) == huella(generar_perfiles(escenario=esc, semilla=s))
            df = a_dataframe(pf)
            acum["p0"].append(df.p0_media.mean())
            acum["s"].append(df.s_media.mean())
            acum["g"].append(df.g_media.mean())
            acum["t"].append(df.t_media.mean())
            acum["abandono"].append(df.abandono.mean())
            acum["lectura1"].append((df.lectura == 1).mean())
            acum["voz"].append(df.voz.mean())
            acum["hipertension"].append(df.hipertension.mean())
        if esc in ("E1",):   # valores objetivo de la Tabla 4.8 en el escenario base
            objetivo = {"s": 0.10, "g": 0.20, "t": 0.15, "abandono": 0.10, "lectura1": 0.30,
                        "voz": 0.7 * 0.3 + 0.3 * 0.7, "hipertension": float(np.mean(P_HIPERTENSION))}
            for k, o in objetivo.items():
                desv[k].append(abs(np.mean(acum[k]) - o))
    return pd.DataFrame([
        {"Comprobación": "Balance de estratos", "Resultado": f"{bal}/{n} poblaciones"},
        {"Comprobación": "Reproducibilidad (huella SHA-256)", "Resultado": f"{rep}/{n} poblaciones"},
        {"Comprobación": "Identificabilidad (s, g <= 0,30)", "Resultado": f"{ident}/{n} poblaciones"},
        {"Comprobación": "Coherencia marginal (escenario base, máx. desviación)",
         "Resultado": f"{max(max(v) for v in desv.values() if v):.4f}"},
        {"Comprobación": "Plausibilidad", "Resultado": "Pendiente: requiere datos reales del piloto"},
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--entrada", default="experimentos/registros/resultados.csv")
    ap.add_argument("--salida", default="experimentos/registros")
    ap.add_argument("--validar-generador", action="store_true")
    a = ap.parse_args()
    out = Path(a.salida)
    (out / "tablas").mkdir(parents=True, exist_ok=True)
    if a.validar_generador:
        t = validar_generador()
        print(t.to_string(index=False))
        t.to_csv(out / "tablas" / "validez_generador.csv", index=False)
    if not Path(a.entrada).exists():
        print(f"No existe {a.entrada}; ejecute primero experimentos.ejecutar")
        return
    df = pd.read_csv(a.entrada)
    df = df[df.parametro.isna() | (df.parametro == "")]
    nsem = df.groupby(["escenario", "politica"]).semilla.nunique().min()
    print(f"{len(df)} ejecuciones; semillas por combinación: {nsem}; "
          f"violaciones de la máscara: {int(df.violaciones.sum())}")
    desc = descriptivos(df)
    contr = contrastes(df)
    desc.to_csv(out / "tablas" / "descriptivos.csv", index=False)
    contr.to_csv(out / "tablas" / "contrastes.csv", index=False)
    t417, t418 = tabla_417(desc, contr), tabla_418(df)
    t417.to_csv(out / "tablas" / "tabla_4_17.csv", index=False)
    t418.to_csv(out / "tablas" / "tabla_4_18.csv", index=False)
    print("\nTabla 4.17 (mejor propuesta frente a la secuencia fija)\n", t417.to_string(index=False))
    print("\nTabla 4.18 (escenario base)\n", t418.to_string(index=False))
    resp, por_esc = evaluar_h1(df, desc, contr)
    print(f"\nH1 {'RESPALDADA' if resp else 'NO respaldada'}; cumple el criterio por escenario: {por_esc}")
    figura_diferencias(df, out / "figuras" / "dif_pareada.png")
    print(f"Figura en {out / 'figuras' / 'dif_pareada.png'}")


if __name__ == "__main__":
    main()
