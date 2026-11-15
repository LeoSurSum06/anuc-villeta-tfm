"""Servicio adaptativo en línea: BKT + política de selección dentro de la máscara de seguridad.

Mantiene por persona el vector de dominio estimado y los contadores de práctica en la base de
datos. La política por defecto es ``reglas_expertas``, la más adecuada para el prototipo según los
resultados de la Subsección 4.4.6; las políticas de bandido (``linucb``, ``thompson``) están
disponibles, pero su estado vive en memoria y se reinicia con el servicio.
"""
import numpy as np

from adaptativo import config
from adaptativo.acciones import mascara_lote
from adaptativo.bkt import bkt_update
from adaptativo.contexto import construir_contexto
from adaptativo.politicas import crear_politica
from adaptativo.recompensa import recompensa

from .bd import BD


class _Ctx:
    """Contexto de una persona con la misma interfaz que usa el simulador (N = 1)."""

    def __init__(self, X, P, adm, voz, lectura, conteo):
        self.X, self.P, self.adm, self.voz, self.lectura, self.conteo = X, P, adm, voz, lectura, conteo


class ServicioAdaptativo:
    def __init__(self, bd: BD, politica="reglas_expertas", semilla=0, version_codigo=""):
        self.bd = bd
        self.nombre_politica = politica
        self.rng = np.random.default_rng(semilla)
        self.politica = crear_politica(politica, self.rng)
        self.version_codigo = version_codigo

    def registrar(self, labor, edad, lectura, voz, diagnostico=None, consiente_sensibles=False,
                  condiciones=None):
        """Registra a una persona. ``diagnostico`` son 14 probabilidades iniciales (opcional);
        sin ellas, el arranque en frío parte de 0,25 en todas las competencias."""
        p0 = diagnostico if diagnostico is not None else [0.25] * config.K
        return self.bd.crear_persona(labor, edad, lectura, voz, p0, consiente_sensibles, condiciones)

    def _contexto(self, pers):
        P = np.array([pers["p_dominio"]])
        cond = pers["condiciones"] if pers["consiente_sensibles"] else {}
        perfil = {"labor": pers["labor"], "edad": pers["edad"], "lectura": pers["lectura"],
                  "voz": pers["voz"], **cond}
        situacion = {"conectividad": 0.9, "franja": 0.5, "dias": 0.0,
                     "sesiones": min(pers["sesiones"] / config.N_RONDAS, 1.0)}
        x = construir_contexto(P[0], perfil, situacion, consentimiento=bool(pers["consiente_sensibles"]))
        lab = np.zeros((1, config.K), dtype=bool)
        lab[0, config.COMPETENCIAS_LABOR[pers["labor"]]] = True
        adm = mascara_lote(P, np.array([pers["voz"]]), lab)
        return _Ctx(x[None, :], P, adm, np.array([pers["voz"]]), np.array([pers["lectura"]]),
                    np.array([pers["conteo"]]))

    def siguiente_actividad(self, codigo):
        pers = self.bd.persona(codigo)
        if pers is None:
            raise KeyError("persona no registrada")
        ctx = self._contexto(pers)
        a = int(self.politica.seleccionar(ctx)[0])
        assert ctx.adm[0, a], "la máscara de seguridad no puede violarse"
        k, tipo, formato, dif = config.ACCIONES[a]
        return {"accion": a, "competencia": f"C{k + 1}", "nombre_competencia": config.COMPETENCIAS[k],
                "tipo": tipo, "formato": formato, "dificultad": dif,
                "contenido_id": f"C{k + 1}-{tipo}-{formato}-{dif}",
                "admisibles": int(ctx.adm.sum())}

    def registrar_respuesta(self, codigo, accion, correcta, completada=True):
        """Actualiza BKT y la política con el resultado de la actividad y devuelve la recompensa."""
        pers = self.bd.persona(codigo)
        ctx = self._contexto(pers)
        k = config.ACCIONES[accion][0]
        p = np.array(pers["p_dominio"])
        p_antes = p[k]
        if completada and correcta is not None:
            s, g, t = (config.BKT_DEFECTO[x] for x in ("s", "g", "t"))
            p[k] = bkt_update(p_antes, bool(correcta), s, g, t)
        r = float(recompensa(p[k] - p_antes, p_antes, 1.0 if completada else 0.0, 0.0 if completada else 1.0))
        conteo = np.array(pers["conteo"])
        conteo[k] += 1
        self.bd.guardar_estado(codigo, p, conteo, pers["sesiones"] + 1)
        self.politica.actualizar(ctx, np.array([accion]), np.array([r]))
        self.bd.registrar_interaccion(codigo, accion, correcta, completada, r, self.nombre_politica,
                                      self.version_codigo)
        return {"recompensa": r, "p_dominio": {f"C{i + 1}": float(v) for i, v in enumerate(p)},
                "dominada": bool(p[k] >= config.DOMINIO)}
