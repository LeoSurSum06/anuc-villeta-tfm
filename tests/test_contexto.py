import numpy as np

from adaptativo import config
from adaptativo.contexto import codificar_perfiles, construir_contexto, contexto_lote
from adaptativo.perfiles import generar_perfiles

SIT = {"conectividad": 0.9, "franja": 0.5, "dias": 0.2, "sesiones": 0.1}


def test_dimension_32():
    pf = generar_perfiles()[0]
    assert construir_contexto(np.full(14, 0.3), pf, SIT).shape == (32,)


def test_categorias_de_referencia_no_activan_ninguna_columna():
    base = {"labor": "Caña/panela", "edad": "<30", "lectura": 1, "voz": 0}
    x = construir_contexto(np.zeros(14), base, SIT)
    assert x[14:18].sum() == 0 and x[18:21].sum() == 0          # labor y edad de referencia
    assert x[-1] == 1.0                                          # sesgo


def test_sin_consentimiento_las_condiciones_valen_cero():
    pf = {"labor": "Café", "edad": ">60", "lectura": 2, "voz": 1, "hipertension": 1, "diabetes": 1}
    con = construir_contexto(np.zeros(14), pf, SIT, consentimiento=True)
    sin = construir_contexto(np.zeros(14), pf, SIT, consentimiento=False)
    assert con[21:25].sum() == 2 and sin[21:25].sum() == 0
    assert np.array_equal(np.delete(con, range(21, 25)), np.delete(sin, range(21, 25)))


def test_version_por_lotes_equivale_a_la_individual():
    pf = generar_perfiles(semilla=3)[:20]
    P = np.random.default_rng(0).random((20, 14))
    sit = {k: np.full(20, v) for k, v in SIT.items()}
    X = contexto_lote(P, codificar_perfiles(pf), sit)
    for i in range(20):
        assert np.allclose(X[i], construir_contexto(P[i], pf[i], SIT))
