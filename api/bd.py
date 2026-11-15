"""Base de datos relacional (SQLite) para perfiles seudonimizados, interacciones y versiones
del corpus. No se guarda nombre, documento de identidad ni teléfono: cada persona se identifica
con un código aleatorio, y la tabla de correspondencia, si se necesita, se custodia por separado.
"""
import json
import secrets
import sqlite3
import time

ESQUEMA = """
CREATE TABLE IF NOT EXISTS personas (
    codigo TEXT PRIMARY KEY, labor TEXT NOT NULL, edad TEXT NOT NULL,
    lectura INTEGER NOT NULL, voz INTEGER NOT NULL,
    consiente_sensibles INTEGER NOT NULL DEFAULT 0,
    condiciones TEXT NOT NULL DEFAULT '{}',
    p_dominio TEXT NOT NULL, conteo TEXT NOT NULL, sesiones INTEGER NOT NULL DEFAULT 0,
    creada REAL NOT NULL);
CREATE TABLE IF NOT EXISTS interacciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT, codigo TEXT NOT NULL, accion INTEGER NOT NULL,
    correcta INTEGER, completada INTEGER NOT NULL, recompensa REAL NOT NULL,
    politica TEXT NOT NULL, version_codigo TEXT, momento REAL NOT NULL,
    FOREIGN KEY (codigo) REFERENCES personas(codigo));
CREATE TABLE IF NOT EXISTS versiones_corpus (
    version TEXT PRIMARY KEY, documentos TEXT NOT NULL, creada REAL NOT NULL);
"""


class BD:
    def __init__(self, ruta=":memory:"):
        self.con = sqlite3.connect(ruta, check_same_thread=False)
        self.con.row_factory = sqlite3.Row
        self.con.executescript(ESQUEMA)

    def crear_persona(self, labor, edad, lectura, voz, p0, consiente_sensibles=False, condiciones=None):
        codigo = secrets.token_hex(6)       # seudónimo aleatorio
        cond = condiciones if consiente_sensibles else {}    # sin consentimiento, no se guardan
        self.con.execute(
            "INSERT INTO personas VALUES (?,?,?,?,?,?,?,?,?,0,?)",
            (codigo, labor, edad, lectura, int(voz), int(consiente_sensibles), json.dumps(cond),
             json.dumps(list(map(float, p0))), json.dumps([0.0] * len(p0)), time.time()))
        self.con.commit()
        return codigo

    def persona(self, codigo):
        f = self.con.execute("SELECT * FROM personas WHERE codigo=?", (codigo,)).fetchone()
        if f is None:
            return None
        d = dict(f)
        d["condiciones"] = json.loads(d["condiciones"])
        d["p_dominio"] = json.loads(d["p_dominio"])
        d["conteo"] = json.loads(d["conteo"])
        return d

    def guardar_estado(self, codigo, p, conteo, sesiones):
        self.con.execute("UPDATE personas SET p_dominio=?, conteo=?, sesiones=? WHERE codigo=?",
                         (json.dumps(list(map(float, p))), json.dumps(list(map(float, conteo))),
                          sesiones, codigo))
        self.con.commit()

    def registrar_interaccion(self, codigo, accion, correcta, completada, recompensa, politica, version_codigo=""):
        self.con.execute(
            "INSERT INTO interacciones (codigo,accion,correcta,completada,recompensa,politica,version_codigo,momento)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (codigo, int(accion), None if correcta is None else int(correcta), int(completada),
             float(recompensa), politica, version_codigo, time.time()))
        self.con.commit()

    def registrar_version_corpus(self, version, documentos):
        self.con.execute("INSERT OR REPLACE INTO versiones_corpus VALUES (?,?,?)",
                         (version, json.dumps(documentos), time.time()))
        self.con.commit()

    def borrar_persona(self, codigo):
        """Derecho de supresión: elimina a la persona y sus interacciones."""
        self.con.execute("DELETE FROM interacciones WHERE codigo=?", (codigo,))
        self.con.execute("DELETE FROM personas WHERE codigo=?", (codigo,))
        self.con.commit()
