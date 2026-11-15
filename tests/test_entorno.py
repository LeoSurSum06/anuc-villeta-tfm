import numpy as np

from adaptativo import config
from adaptativo.entorno import Entorno, simular
from adaptativo.perfiles import generar_perfiles


def test_la_mascara_no_se_viola_en_ninguna_politica_ni_escenario():
    for esc in config.ESCENARIOS:
        pf = generar_perfiles(escenario=esc, semilla=1)
        for pol in config.POLITICAS:
            assert simular(pf, esc, 1, pol, n_rondas=15)["violaciones"] == 0


def test_reproducibilidad_de_la_simulacion():
    pf = generar_perfiles(semilla=2)
    assert simular(pf, "E1", 2, "linucb", n_rondas=20) == simular(pf, "E1", 2, "linucb", n_rondas=20)


def test_el_oraculo_no_tiene_regret_y_supera_a_la_aleatoria():
    pf = generar_perfiles(semilla=0)
    o = simular(pf, "E1", 0, "oraculo", n_rondas=60)
    a = simular(pf, "E1", 0, "aleatoria", n_rondas=60)
    assert abs(o["regret"]) < 1e-9
    assert o["recompensa_eval"] > a["recompensa_eval"]


def test_las_politicas_comparten_los_mismos_numeros_aleatorios():
    pf = generar_perfiles(semilla=0)
    e1, e2 = Entorno(pf, "E1", 0, 5), Entorno(pf, "E1", 0, 5)
    assert np.array_equal(e1.z, e2.z)
    c1, c2 = e1.estado(), e2.estado()
    assert np.array_equal(e1._u["resp"], e2._u["resp"])


def test_arranque_en_frio_parte_de_025():
    env = Entorno(generar_perfiles(escenario="E2", semilla=0), "E2", 0, 5)
    assert np.all(env.P == 0.25)


def test_el_olvido_solo_ocurre_en_e6():
    pf = generar_perfiles(semilla=0)
    for esc, debe_olvidar in (("E1", False), ("E6", True)):
        env = Entorno(generar_perfiles(escenario=esc, semilla=0), esc, 0, 60)
        env.z[:] = True
        perdidos = 0
        for _ in range(60):
            env.estado()
            perdidos += int((~env.z).sum())
            env.z[:] = True
        assert (perdidos > 0) == debe_olvidar
