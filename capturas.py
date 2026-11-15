"""Genera las capturas y el resumen estadístico de la Sección 4.3 (Figuras 4.5 a 4.7, Tablas 4.9 y 4.10).

Usa el generador del repositorio con la semilla 0 y el escenario base y escribe en
``experimentos/registros/figuras`` las imágenes de la consola de pandas (``df.head()`` y
``describe()``) y el mapa de calor del dominio inicial por labor y competencia.

    python -m experimentos.capturas
"""
import os
from pathlib import Path

SALIDA = Path("experimentos/registros/figuras")
SALIDA.mkdir(parents=True, exist_ok=True)
os.chdir(SALIDA)   # las rutas relativas del script escriben aquí

import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from adaptativo.config import K, LABORES
from adaptativo.perfiles import a_dataframe, generar_perfiles

df = a_dataframe(generar_perfiles(escenario="E1", semilla=0))
df.to_pickle("df_seed0.pkl")

# --- Captura 1: df.head() estilo consola
cols = ["id","labor","edad","hipertension","lectura","voz","abandono","p0_media","s_media","g_media","t_media"]
head = df[cols].head(8).copy()
for c in ["abandono","p0_media","s_media","g_media","t_media"]:
    head[c] = head[c].round(3)
txt = ">>> df = a_dataframe(generar_perfiles(escenario='E1', semilla=0))\n>>> df[cols].head(8)\n" + head.to_string()
txt += f"\n\n>>> df.shape\n{df.shape}"
def consola(txt, fn, w=11.5):
    lines = txt.split("\n")
    h = 0.20*len(lines)+0.3
    fig = plt.figure(figsize=(w,h), dpi=200)
    fig.patch.set_facecolor("#F2F4F7")
    fig.text(0.015, 0.97, txt, family="DejaVu Sans Mono", fontsize=8.2, va="top", ha="left", color="#1b1f24", linespacing=1.25)
    fig.savefig(fn, facecolor=fig.get_facecolor()); plt.close(fig)
consola(txt, "fig_df_head.png")

# --- Resumen estadistico: media, moda, desviacion
num = ["p0_media","s_media","g_media","t_media","abandono"]
res = pd.DataFrame({
    "Media": df[num].mean(),
    "Moda": [df[c].round(2).mode().iloc[0] for c in num],
    "Desv. est.": df[num].std(),
    "Mín.": df[num].min(), "Máx.": df[num].max()}).round(3)
res.to_pickle("res_num.pkl")
cat = {}
for c in ["labor","edad","lectura","voz","hipertension"]:
    vc = df[c].value_counts()
    cat[c] = (df[c].mode().iloc[0], int(vc.max()), len(vc))
print(res); print(cat)
print(df.groupby("labor")["p0_media"].agg(["mean","std"]).round(3))
print(df.groupby("edad")["hipertension"].mean().round(2))
print(df["lectura"].value_counts(normalize=True).sort_index().round(2), df["voz"].mean())

desc = df[num].describe().round(3).to_string()
consola(">>> df[['p0_media','s_media','g_media','t_media','abandono']].describe()\n"+desc, "fig_df_describe.png", w=8.5)

# --- Heatmap p0 medio por labor x competencia (misma semilla)
M = np.array([[np.mean([r[f"p0_C{k+1}"] for _,r in df[df.labor==l].iterrows()]) for k in range(K)] for l in LABORES])
fig, ax = plt.subplots(figsize=(10,3.6), dpi=180)
im = ax.imshow(M, cmap="Blues", vmin=0, vmax=0.6, aspect="auto")
ax.set_xticks(range(K)); ax.set_xticklabels([f"C{k+1}" for k in range(K)])
ax.set_yticks(range(5)); ax.set_yticklabels(LABORES)
for i in range(5):
    for j in range(K):
        ax.text(j,i,f"{M[i,j]:.2f}".replace(".",","),ha="center",va="center",fontsize=8,color="white" if M[i,j]>0.33 else "black")
ax.set_xlabel("Competencia"); ax.set_ylabel("Labor principal")
cb = fig.colorbar(im, ax=ax); cb.set_label("p0 media")
fig.tight_layout(); fig.savefig("fig_heat.png"); plt.close(fig)
