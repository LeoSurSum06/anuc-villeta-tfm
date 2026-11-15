import numpy as np

from adaptativo import config
from adaptativo.acciones import mascara_lote
from adaptativo.entorno import Contexto, Entorno
from adaptativo.perfiles import generar_perfiles
from adaptativo.politicas import LinUCB, ThompsonLineal, crear_politica


def _ctx(env):
    return env.estado()


def test_todas_las_politicas_eligen_solo_acciones_admisibles():
    pf = generar_perfiles(semilla=0)
    for nombre in config.POLITICAS:
        env = Entorno(pf, "E1", 0, n_rondas=6)
        pol = crear_politica(nombre, np.random.default_rng(0), entorno=env)
        for _ in range(6):
            ctx = env.estado()
            acc = pol.seleccionar(ctx)
            assert ctx.adm[np.arange(len(acc)), acc].all(), nombre
            pol.actualizar(ctx, acc, env.paso(ctx, acc))
        assert env.violaciones == 0


def test_sherman_morrison_equivale_a_la_inversa_directa():
    rng = np.random.default_rng(0)
    pol = LinUCB(rng, lam=1.0)
    A = np.eye(config.D_CONTEXTO)
    b = np.zeros(config.D_CONTEXTO)
    for _ in range(50):
        x = rng.random(config.D_CONTEXTO)
        r = rng.random()
        pol._sherman_morrison(3, x, r)
        A += np.outer(x, x)
        b += r * x
    assert np.allclose(pol.A_inv[3], np.linalg.inv(A))
    assert np.allclose(pol.theta[3], np.linalg.solve(A, b))
    assert np.allclose(pol.A_inv[4], np.eye(config.D_CONTEXTO))      # las demás no cambian


def test_los_bandidos_aprenden_una_recompensa_lineal():
    rng = np.random.default_rng(1)
    w = rng.normal(size=config.D_CONTEXTO) * 0.1
    pol = ThompsonLineal(rng, v=0.1)
    for _ in range(2000):
        x = rng.random(config.D_CONTEXTO)
        pol._sherman_morrison(0, x, float(x @ w))
    assert np.allclose(pol.theta[0], w, atol=0.05)


def test_la_exploracion_de_linucb_disminuye_con_las_observaciones():
    rng = np.random.default_rng(2)
    pol = LinUCB(rng)
    x = rng.random(config.D_CONTEXTO)
    _, v0 = pol._media_var(np.array([0]), x)
    for _ in range(30):
        pol._sherman_morrison(0, x, 0.3)
    _, v1 = pol._media_var(np.array([0]), x)
    assert v1[0] < v0[0]
