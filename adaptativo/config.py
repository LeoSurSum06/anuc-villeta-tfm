"""Configuración central del motor adaptativo y del simulador.

Reúne en un único lugar las constantes que definen el problema: las 14
competencias, el espacio de 336 acciones, los pesos de la recompensa
(Ecuación 4.4), los parámetros por defecto de BKT y de las políticas, y la
definición de los seis escenarios de simulación (Tabla 4.12).

Todos los valores son parámetros de diseño fijados por juicio experto, tal y
como se declara en la memoria; se recalibrarán con los datos del piloto de campo.
"""
from itertools import product

# --------------------------------------------------------------------------
# Competencias (Tabla 4.5). Índices 0-based: C1 -> 0, ..., C14 -> 13.
# --------------------------------------------------------------------------
COMPETENCIAS = [
    "Prevención en el uso de herramientas y del trapiche",      # C1
    "Hidratación y exposición al calor",                        # C2
    "Ergonomía y manipulación de cargas",                       # C3
    "Heridas y hemorragias",                                    # C4
    "Picaduras y mordeduras",                                   # C5
    "Quemaduras",                                               # C6
    "Reconocimiento de señales de alarma y activación de emergencias",  # C7
    "Manejo seguro de agroquímicos",                            # C8
    "Seguridad del visitante en senderos y fincas",             # C9
    "Atención al visitante",                                    # C10
    "Diseño de experiencias y relatos de la cultura panelera",  # C11
    "Precio y presentación de productos",                       # C12
    "Prácticas ambientales",                                    # C13
    "Formalización y normativa básica",                         # C14
]
K = len(COMPETENCIAS)          # 14
C7 = 6                         # señales de alarma (índice 0-based)
C4_C6 = (3, 4, 5)              # heridas, picaduras y quemaduras

# --------------------------------------------------------------------------
# Población (Tabla 4.8 y Tabla 4.6)
# --------------------------------------------------------------------------
LABORES = ["Caña/panela", "Café", "Frutales", "Ganadería", "Turismo"]
EDADES = ["<30", "30-45", "46-60", ">60"]
COMPETENCIAS_LABOR = {
    "Caña/panela": [0, 1, 2, 5],            # C1, C2, C3, C6
    "Café":        [0, 1, 2, 7],            # C1, C2, C3, C8
    "Frutales":    [0, 2, 4, 7],            # C1, C3, C5, C8
    "Ganadería":   [2, 3, 4],               # C3, C4, C5
    "Turismo":     [8, 9, 10, 11, 12, 13],  # C9 a C14
}
CONDICIONES = ["hipertension", "diabetes", "osteomuscular", "embarazo"]

# --------------------------------------------------------------------------
# Espacio de acciones: 14 x 4 x 2 x 3 = 336
# --------------------------------------------------------------------------
TIPOS = ("microleccion", "caso_practico", "pregunta_evaluacion", "repaso_espaciado")
FORMATOS = ("texto_ilustracion", "audio")
DIFICULTADES = ("basica", "intermedia", "avanzada")
ACCIONES = [(k, tipo, formato, dif)
            for k, tipo, formato, dif in product(range(K), TIPOS, FORMATOS, DIFICULTADES)]
N_ACCIONES = len(ACCIONES)     # 336
assert N_ACCIONES == 336

# --------------------------------------------------------------------------
# Modelo BKT (Ecuaciones 4.1 a 4.3) y máscara de seguridad
# --------------------------------------------------------------------------
BKT_DEFECTO = {"s": 0.10, "g": 0.20, "t": 0.15}  # parámetros que el sistema supone
COTA_SG = 0.30            # identificabilidad: s <= 0,30 y g <= 0,30
UMBRAL_C7 = 0.6           # regla 1: prioridad obligatoria de C7
DOMINIO = 0.95            # umbral de dominio de una competencia

# --------------------------------------------------------------------------
# Contexto (Tabla 4.6): d = 32
# --------------------------------------------------------------------------
D_CONTEXTO = K + (len(LABORES) - 1) + (len(EDADES) - 1) + len(CONDICIONES) + 2 + 4 + 1
assert D_CONTEXTO == 32

# --------------------------------------------------------------------------
# Recompensa (Ecuación 4.4): r = clip(alfa*dp/(1-p) + beta*f_int - gamma*aband, 0, 1)
# --------------------------------------------------------------------------
PESOS_RECOMPENSA = {"alfa": 0.6, "beta": 0.3, "gamma": 0.1}

# --------------------------------------------------------------------------
# Políticas (Ecuaciones 4.5 a 4.7). Valores por defecto del análisis de sensibilidad.
# --------------------------------------------------------------------------
LAMBDA_RIDGE = 1.0
ALPHA_UCB = 1.0
V_THOMPSON = 0.3
EPSILON = 0.1
POLITICAS = ("secuencia_fija", "aleatoria", "reglas_expertas", "epsilon_greedy",
             "oraculo", "linucb", "thompson")

# --------------------------------------------------------------------------
# Simulación
# --------------------------------------------------------------------------
N_PERFILES = 100          # 5 labores x 4 rangos de edad x 5 perfiles
N_RONDAS = 200            # interacciones por perfil
SEMILLAS = tuple(range(50))   # 0 a 49
DIAS_ENTRE_SESIONES = (1, 1, 1, 2, 3)   # muestreo uniforme de los saltos entre sesiones
OLVIDO_DIAS = 14          # E6: días sin práctica antes de empezar a olvidar
OLVIDO_PROB_DIA = 0.05    # E6: probabilidad diaria de perder el dominio

ESCENARIOS = {
    "E1": "Base",
    "E2": "Arranque en frío",
    "E3": "Conectividad baja",
    "E4": "Heterogeneidad alta",
    "E5": "Ruido",
    "E6": "No estacionariedad",
}

# Versión de la configuración (se registra en cada ejecución)
VERSION_CONFIG = "1.0"
