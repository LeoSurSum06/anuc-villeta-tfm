import numpy as np
import pytest

from adaptativo.perfiles import a_dataframe, generar_perfiles, huella, validar_poblacion


def test_poblacion_de_100_perfiles_y_26_columnas():
    df = a_dataframe(generar_perfiles(semilla=0))
    assert df.shape == (100, 26)


def test_reproducibilidad_misma_semilla_misma_huella():
    assert huella(generar_perfiles(escenario="E4", semilla=7)) == huella(generar_perfiles(escenario="E4", semilla=7))
    assert huella(generar_perfiles(semilla=1)) != huella(generar_perfiles(semilla=2))


def test_balance_de_estratos_e_identificabilidad_en_todos_los_escenarios():
    for esc in ("E1", "E2", "E3", "E4", "E5", "E6"):
        for s in (0, 1, 2):
            v = validar_poblacion(generar_perfiles(escenario=esc, semilla=s))
            assert v["balance_estratos"] and v["identificabilidad"]


def test_escenarios_modifican_solo_lo_declarado():
    base = generar_perfiles(semilla=0)
    e3 = generar_perfiles(escenario="E3", semilla=0)
    assert all(p["p_corte"] == 0.30 for p in e3) and all(p["p_corte"] == 0.05 for p in base)
    e5 = generar_perfiles(escenario="E5", semilla=0)
    assert all(np.all(p["s"] == 0.25) and np.all(p["g"] == 0.25) for p in e5)
    var = lambda pf: np.mean([p["p0"].var() for p in pf])
    assert var(generar_perfiles(escenario="E4", semilla=0)) > var(base)


def test_la_experiencia_laboral_suma_015_con_tope_09():
    for p in generar_perfiles(semilla=5):
        assert p["p0"].max() <= 0.9 + 1e-12


def test_valores_de_la_tabla_4_9_con_semilla_0():
    d = a_dataframe(generar_perfiles(escenario="E1", semilla=0))
    assert d.p0_media.mean() == pytest.approx(0.291, abs=5e-4)
    assert d.s_media.mean() == pytest.approx(0.101, abs=5e-4)
    assert d.g_media.mean() == pytest.approx(0.192, abs=5e-4)
    assert d.t_media.mean() == pytest.approx(0.154, abs=5e-4)
    assert d.abandono.mean() == pytest.approx(0.099, abs=5e-4)
    assert d.hipertension.mean() == pytest.approx(0.15)


def test_las_condiciones_adicionales_no_alteran_la_poblacion_base():
    ref = a_dataframe(generar_perfiles(semilla=0))
    comp = a_dataframe(generar_perfiles(semilla=0), completo=True)
    assert comp.shape == (100, 29)
    assert ref.equals(comp.drop(columns=["diabetes", "osteomuscular", "embarazo"]))
