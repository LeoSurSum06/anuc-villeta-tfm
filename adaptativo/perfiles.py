"""Generador de perfiles sintéticos (Sección 4.3).

Produce, para un escenario (E1 a E6) y una semilla, una población estratificada de
100 perfiles de campesinos aprendices: 5 labores x 4 rangos de edad x 5 perfiles.

Los atributos observables (labor, edad, condiciones, lectura y voz) son los que el
sistema conocería tras el registro; los ocultos (parámetros BKT verdaderos,
propensión al abandono y corte de sesión) los usa solo el simulador.

Reproducibilidad: el mismo par (escenario, semilla) produce siempre la misma
población. Las tres condiciones de salud adicionales a la hipertensión se generan
con un flujo aleatorio independiente, de modo que no alteran la población descrita
en la Tabla 4.8 ni las capturas del Anexo B.
"""
import hashlib

import numpy as np
import pandas as pd

from .bkt import parametros_identificables
from .config import COMPETENCIAS_LABOR, EDADES, K, LABORES

# Probabilidades de cada condición por rango de edad (<30, 30-45, 46-60, >60).
# Supuestos de simulación fijados por juicio experto; no son estimaciones
# epidemiológicas. Solo la hipertensión figura en la Tabla 4.8 de la memoria;
# las otras tres son supuestos del repositorio y pueden editarse aquí.
P_HIPERTENSION = [0.02, 0.08, 0.20, 0.35]
P_DIABETES = [0.01, 0.04, 0.10, 0.16]
P_OSTEOMUSCULAR = [0.03, 0.10, 0.25, 0.40]
P_EMBARAZO = [0.03, 0.02, 0.0, 0.0]
_SALT_CONDICIONES = 7919


def beta_media(rng, mu, kappa, n):
    """Beta(mu*kappa, (1-mu)*kappa): media mu y concentración kappa."""
    return rng.beta(mu * kappa, (1.0 - mu) * kappa, size=n)


def generar_perfiles(n_por_estrato=5, escenario="E1", semilla=0):
    """Genera la población sintética. Devuelve una lista de diccionarios."""
    rng = np.random.default_rng(semilla)               # reproducible
    conc = 10.0 if escenario == "E4" else 20.0         # heterogeneidad
    p_corte = 0.30 if escenario == "E3" else 0.05      # corte de sesión
    perfiles = []
    for labor in LABORES:                              # 5 labores
        for i_edad, edad in enumerate(EDADES):         # x 4 rangos = 20 estratos
            for _ in range(n_por_estrato):             # x 5 perfiles
                p0 = beta_media(rng, 0.25, conc, K)    # dominio inicial verdadero
                idx = COMPETENCIAS_LABOR[labor]        # experiencia en su labor
                p0[idx] = np.minimum(0.9, p0[idx] + 0.15)
                s = beta_media(rng, 0.10, 20.0, K).clip(0.01, 0.30)   # desliz
                g = beta_media(rng, 0.20, 20.0, K).clip(0.01, 0.30)   # adivinanza
                if escenario == "E5":                  # ruido elevado
                    s[:], g[:] = 0.25, 0.25
                t = beta_media(rng, 0.15, conc, K)     # aprendizaje
                abandono = beta_media(rng, 0.10, conc, 1)[0]
                lectura = int(rng.choice([1, 2, 3], p=[0.3, 0.4, 0.3]))
                voz = int(rng.random() < (0.7 if lectura == 1 else 0.3))
                hipertension = int(rng.random() < P_HIPERTENSION[i_edad])
                perfiles.append({"labor": labor, "edad": edad, "i_edad": i_edad,
                                 "hipertension": hipertension,
                                 "lectura": lectura, "voz": voz,
                                 "abandono": abandono, "p_corte": p_corte,
                                 "p0": p0, "s": s, "g": g, "t": t})
    # Otras condiciones declaradas: flujo aleatorio independiente
    rc = np.random.default_rng([semilla, _SALT_CONDICIONES])
    for pf in perfiles:
        i = pf["i_edad"]
        pf["diabetes"] = int(rc.random() < P_DIABETES[i])
        pf["osteomuscular"] = int(rc.random() < P_OSTEOMUSCULAR[i])
        pf["embarazo"] = int(rc.random() < P_EMBARAZO[i])
    return perfiles


def a_dataframe(perfiles, completo=False):
    """Aplana los perfiles en un DataFrame (una fila por perfil).

    Por defecto devuelve las 26 columnas del Anexo B. Con ``completo=True`` añade
    las otras tres condiciones de salud declaradas.
    """
    filas = []
    for i, p in enumerate(perfiles):
        fila = {"id": i, "labor": p["labor"], "edad": p["edad"],
                "hipertension": p["hipertension"], "lectura": p["lectura"],
                "voz": p["voz"], "abandono": p["abandono"],
                "p_corte": p["p_corte"]}
        if completo:
            fila.update({c: p[c] for c in ("diabetes", "osteomuscular", "embarazo")})
        for k in range(K):
            fila[f"p0_C{k + 1}"] = p["p0"][k]
        fila["p0_media"] = p["p0"].mean()
        fila["s_media"] = p["s"].mean()
        fila["g_media"] = p["g"].mean()
        fila["t_media"] = p["t"].mean()
        filas.append(fila)
    return pd.DataFrame(filas)


def huella(perfiles):
    """Huella SHA-256 del contenido de la población (comprobación de reproducibilidad)."""
    h = hashlib.sha256()
    for p in perfiles:
        h.update(f"{p['labor']}|{p['edad']}|{p['hipertension']}|{p['diabetes']}|"
                 f"{p['osteomuscular']}|{p['embarazo']}|{p['lectura']}|{p['voz']}|".encode())
        for clave in ("abandono", "p_corte"):
            h.update(np.float64(p[clave]).tobytes())
        for clave in ("p0", "s", "g", "t"):
            h.update(np.ascontiguousarray(p[clave], dtype=np.float64).tobytes())
    return h.hexdigest()


def validar_poblacion(perfiles, n_por_estrato=5):
    """Comprobaciones de validez de la Subsección 4.3.4 que dependen solo de la población.

    La reproducibilidad y la coherencia marginal requieren varias ejecuciones y se
    comprueban en ``tests/test_perfiles.py`` y en ``experimentos/analizar.py``.
    """
    conteo = pd.Series([(p["labor"], p["edad"]) for p in perfiles]).value_counts()
    balance = (len(conteo) == len(LABORES) * len(EDADES)
               and bool((conteo == n_por_estrato).all()))
    ident = all(parametros_identificables(p["s"], p["g"]) for p in perfiles)
    return {"balance_estratos": balance, "identificabilidad": ident}


if __name__ == "__main__":
    df = a_dataframe(generar_perfiles(escenario="E1", semilla=0))
    pd.set_option("display.width", 120)
    print(df.head())
    print(df.describe().round(3))
