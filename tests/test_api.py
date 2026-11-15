import numpy as np
import pytest

from adaptativo import config
from api.bd import BD
from api.orquestador import Orquestador
from api.servicios import ServicioAdaptativo
from voz.stt import wer


@pytest.fixture
def orq():
    return Orquestador(ServicioAdaptativo(BD(":memory:")))


def _registrar(orq, **kw):
    datos = dict(labor="Café", edad="30-45", lectura=2, voz=False)
    datos.update(kw)
    return orq.registrar(**datos)["codigo"]


def test_la_primera_actividad_es_de_c7_por_la_prioridad_obligatoria(orq):
    cod = _registrar(orq)
    act = orq.siguiente_actividad(cod)
    assert act["competencia"] == "C7" and act["admisibles"] == 24


def test_registrar_una_respuesta_actualiza_bkt_y_el_estado(orq):
    cod = _registrar(orq)
    act = orq.siguiente_actividad(cod)
    r = orq.registrar_respuesta(cod, act["accion"], True)
    assert r["p_dominio"]["C7"] > 0.25 and 0.0 <= r["recompensa"] <= 1.0
    r2 = orq.registrar_respuesta(cod, act["accion"], False)
    assert r2["p_dominio"]["C7"] < r["p_dominio"]["C7"]


def test_el_servicio_nunca_viola_la_mascara_en_una_sesion_larga(orq):
    cod = _registrar(orq, voz=True, labor="Turismo")
    rng = np.random.default_rng(0)
    for _ in range(60):
        act = orq.siguiente_actividad(cod)
        assert act["formato"] == "audio"
        orq.registrar_respuesta(cod, act["accion"], bool(rng.random() < 0.7))


def test_sin_consentimiento_no_se_guardan_las_condiciones_de_salud(orq):
    cod = _registrar(orq, consiente_sensibles=False, condiciones={"hipertension": 1})
    assert orq.adaptativo.bd.persona(cod)["condiciones"] == {}
    cod2 = _registrar(orq, consiente_sensibles=True, condiciones={"hipertension": 1})
    assert orq.adaptativo.bd.persona(cod2)["condiciones"] == {"hipertension": 1}


def test_la_emergencia_se_atiende_aunque_no_haya_asistente(orq):
    r = orq.consultar("mi hijo no puede respirar")
    assert r["tipo"] == "emergencia" and "123" in r["texto"]


def test_la_voz_exige_confirmacion_antes_de_ejecutar():
    class Falso:
        def transcribir(self, audio, idioma="es"):
            return "como cuido una quemadura", 0.4

    o = Orquestador(ServicioAdaptativo(BD(":memory:")), transcriptor=Falso())
    r = o.consultar_por_voz("audio.wav")
    assert r["requiere_confirmacion"] and r["confianza_baja"] and "tipo" not in r


def test_la_transcripcion_con_alarma_se_atiende_de_inmediato():
    class Falso:
        def transcribir(self, audio, idioma="es"):
            return "se ahoga ayuda", 0.9

    o = Orquestador(ServicioAdaptativo(BD(":memory:")), transcriptor=Falso())
    assert o.consultar_por_voz("a.wav")["tipo"] == "emergencia"


def test_ruta_recomendada_por_el_orquestador(orq):
    from rutas.optw import generar_instancia
    i = generar_instancia(8, 0)
    r = orq.ruta(i.coords.tolist(), i.puntuacion.tolist(), i.duracion.tolist(), i.ventana.tolist(), i.t_max)
    assert r["ruta"][0] == 0 and r["ruta"][-1] == 9 and r["puntuacion"] > 0


def test_supresion_de_datos_de_la_persona(orq):
    cod = _registrar(orq)
    orq.adaptativo.bd.borrar_persona(cod)
    with pytest.raises(KeyError):
        orq.siguiente_actividad(cod)


def test_wer():
    assert wer("la quemadura se enfría con agua", "la quemadura se enfría con agua") == 0
    assert wer("a b c d", "a b x d") == 0.25
    assert wer("a b", "") == 1.0
