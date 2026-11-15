import numpy as np

from adaptativo import config
from adaptativo.acciones import mascara, mascara_lote, mascara_np

PERFIL = {"voz": 0}


def estado(p7, base=0.3):
    p = np.full(config.K, base)
    p[config.C7] = p7
    return p


def test_el_espacio_tiene_336_acciones():
    assert config.N_ACCIONES == 14 * 4 * 2 * 3 == 336


def test_regla_1_prioridad_de_c7_con_voz_deja_12_acciones():
    adm = mascara(estado(0.3), {"voz": 1}, config.COMPETENCIAS_LABOR["Café"])
    assert len(adm) == 12
    assert all(config.ACCIONES[a][0] == config.C7 and config.ACCIONES[a][2] == "audio" for a in adm)


def test_regla_1_sin_voz_solo_c7():
    adm = mascara(estado(0.3), PERFIL, config.COMPETENCIAS_LABOR["Café"])
    assert {config.ACCIONES[a][0] for a in adm} == {config.C7} and len(adm) == 24


def test_regla_2_sin_avanzadas_de_c4_a_c6_si_c7_no_esta_dominada():
    adm = mascara(estado(0.8), PERFIL, config.COMPETENCIAS_LABOR["Ganadería"])   # C3, C4, C5
    for a in adm:
        k, _, _, dif = config.ACCIONES[a]
        assert not (k in config.C4_C6 and dif == "avanzada")
    adm = mascara(estado(0.96), PERFIL, config.COMPETENCIAS_LABOR["Ganadería"])
    assert any(config.ACCIONES[a][3] == "avanzada" and config.ACCIONES[a][0] in config.C4_C6 for a in adm)


def test_regla_3_con_voz_no_hay_acciones_solo_textuales():
    adm = mascara(estado(0.8), {"voz": 1}, config.COMPETENCIAS_LABOR["Turismo"])
    assert adm and all(config.ACCIONES[a][2] == "audio" for a in adm)


def test_regla_4_prioriza_competencias_de_la_labor_no_dominadas():
    p = estado(0.8)
    adm = mascara(p, PERFIL, config.COMPETENCIAS_LABOR["Turismo"])
    assert {config.ACCIONES[a][0] for a in adm} <= set(config.COMPETENCIAS_LABOR["Turismo"])
    p[8:14] = 0.99       # todas dominadas: la regla suave mantiene todas las admisibles
    assert len(mascara(p, PERFIL, config.COMPETENCIAS_LABOR["Turismo"])) > 24 * 6 // 2


def test_ejemplo_de_88_acciones_admisibles():
    # Caña/panela, sin voz, C7 entre 0,6 y 0,95: C6 pierde sus 8 acciones avanzadas
    assert len(mascara(estado(0.8), PERFIL, config.COMPETENCIAS_LABOR["Caña/panela"])) == 88


def test_nunca_queda_vacio_y_las_versiones_son_equivalentes():
    rng = np.random.default_rng(0)
    labores = list(config.COMPETENCIAS_LABOR)
    N = 300
    P = rng.random((N, config.K))
    P[rng.random(N) < 0.3, config.C7] = rng.uniform(0.9, 1.0)
    voz = rng.random(N) < 0.5
    lab = np.zeros((N, config.K), dtype=bool)
    nombres = [labores[i % 5] for i in range(N)]
    for i, l in enumerate(nombres):
        lab[i, config.COMPETENCIAS_LABOR[l]] = True
    lote = mascara_lote(P, voz.astype(float), lab)
    for i in range(N):
        ref = mascara(P[i], {"voz": int(voz[i])}, config.COMPETENCIAS_LABOR[nombres[i]])
        assert len(ref) > 0
        assert sorted(ref) == sorted(np.flatnonzero(lote[i]).tolist())
        assert sorted(ref) == sorted(mascara_np(P[i], int(voz[i]), config.COMPETENCIAS_LABOR[nombres[i]]).tolist())
