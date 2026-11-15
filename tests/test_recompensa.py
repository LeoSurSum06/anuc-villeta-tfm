import numpy as np
import pytest

from adaptativo.recompensa import recompensa, recompensa_evaluacion


def test_ecuacion_4_4():
    # 0,6 * 0,2/0,5 + 0,3 * 1 - 0,1 * 0 = 0,54
    assert recompensa(0.2, 0.5, 1, 0) == pytest.approx(0.54)


def test_penaliza_el_abandono_y_se_acota_en_cero_uno():
    assert recompensa(0.0, 0.5, 0, 1) == 0.0                     # -0,1 se recorta a 0
    assert recompensa(0.9, 0.5, 1, 0) == 1.0                     # supera 1: se recorta
    rng = np.random.default_rng(0)
    r = recompensa(rng.normal(size=1000), rng.random(1000), rng.integers(0, 2, 1000), rng.integers(0, 2, 1000))
    assert r.min() >= 0 and r.max() <= 1


def test_la_recompensa_de_evaluacion_usa_la_ganancia_latente():
    assert recompensa_evaluacion(1.0, 0.25, 1, 0) == 1.0
    assert recompensa_evaluacion(0.0, 0.25, 1, 0) == pytest.approx(0.3)


def test_no_divide_por_cero_con_p_igual_a_uno():
    assert np.isfinite(recompensa(0.01, 1.0, 1, 0))
