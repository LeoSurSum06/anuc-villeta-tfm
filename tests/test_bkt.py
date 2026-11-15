import numpy as np
import pytest

from adaptativo.bkt import bkt_update, bkt_update_vec, dominada, parametros_identificables


def test_respuesta_correcta_ecuaciones_4_1_y_4_3():
    p, s, g, t = 0.3, 0.1, 0.2, 0.15
    post = p * (1 - s) / (p * (1 - s) + (1 - p) * g)
    assert bkt_update(p, True, s, g, t) == pytest.approx(post + (1 - post) * t)


def test_respuesta_incorrecta_ecuaciones_4_2_y_4_3():
    p, s, g, t = 0.3, 0.1, 0.2, 0.15
    post = p * s / (p * s + (1 - p) * (1 - g))
    assert bkt_update(p, False, s, g, t) == pytest.approx(post + (1 - post) * t)


def test_acierto_sube_y_fallo_baja_la_probabilidad_antes_del_aprendizaje():
    p, s, g = 0.5, 0.1, 0.2
    assert bkt_update(p, True, s, g, 0.0) > p > bkt_update(p, False, s, g, 0.0)


def test_la_probabilidad_se_mantiene_en_cero_uno():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        p = rng.random()
        for c in (True, False):
            assert 0.0 <= bkt_update(p, c, rng.uniform(0.01, 0.3), rng.uniform(0.01, 0.3), rng.random()) <= 1.0


def test_version_vectorizada_equivale_a_la_escalar():
    rng = np.random.default_rng(1)
    p = rng.random(50)
    c = rng.random(50) < 0.5
    esperado = [bkt_update(pi, ci, 0.1, 0.2, 0.15) for pi, ci in zip(p, c)]
    assert np.allclose(bkt_update_vec(p, c, 0.1, 0.2, 0.15), esperado)


def test_identificabilidad():
    assert parametros_identificables([0.1, 0.3], [0.2, 0.3])
    assert not parametros_identificables([0.35], [0.1])
    assert not parametros_identificables([0.1], [0.31])


def test_umbral_de_dominio():
    assert dominada(0.95) and not dominada(0.949)
