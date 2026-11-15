import itertools

import numpy as np

from rutas.optw import (Instancia, generar_instancia, puntuacion_ruta, resolver_exacto,
                        resolver_heuristica, simular_ruta)


def _fuerza_bruta(inst):
    T = inst.tiempos()
    n = inst.n
    mejor = 0.0
    for k in range(n + 1):
        for perm in itertools.permutations(range(1, n + 1), k):
            ruta = [0, *perm, n + 1]
            if simular_ruta(inst, T, ruta)[0]:
                mejor = max(mejor, puntuacion_ruta(inst, ruta))
    return mejor


def test_el_exacto_coincide_con_la_fuerza_bruta():
    for semilla in range(5):
        inst = generar_instancia(6, semilla)
        assert resolver_exacto(inst)[1] == _fuerza_bruta(inst)


def test_la_heuristica_es_factible_y_no_supera_al_optimo():
    for semilla in range(10):
        inst = generar_instancia(8 + semilla % 8, semilla)
        T = inst.tiempos()
        ruta, p = resolver_heuristica(inst, semilla)
        assert simular_ruta(inst, T, ruta)[0]
        assert p <= resolver_exacto(inst)[1] + 1e-9
        assert ruta[0] == 0 and ruta[-1] == inst.n + 1 and len(set(ruta)) == len(ruta)


def test_respeta_el_tiempo_disponible_y_las_ventanas():
    inst = generar_instancia(10, 3)
    inst.t_max = 30.0
    inst.ventana[-1, 1] = 30.0
    ruta, _ = resolver_heuristica(inst)
    T = inst.tiempos()
    ok, t_fin = simular_ruta(inst, T, ruta)
    assert ok and t_fin <= 30.0 + 1e-9


def test_los_tramos_inseguros_se_excluyen_del_grafo():
    inst = generar_instancia(8, 4)
    ruta, _ = resolver_heuristica(inst)
    inst.inseguros = {(a, b) for a, b in zip(ruta[:-1], ruta[1:])}     # se bloquean todos sus tramos
    nueva, _ = resolver_heuristica(inst)
    assert not any((a, b) in inst.inseguros for a, b in zip(nueva[:-1], nueva[1:]))
    nueva_e, _ = resolver_exacto(inst)
    assert not any((a, b) in inst.inseguros for a, b in zip(nueva_e[:-1], nueva_e[1:]))


def test_determinismo_con_semilla():
    inst = generar_instancia(12, 1)
    assert resolver_heuristica(inst, 5) == resolver_heuristica(inst, 5)
