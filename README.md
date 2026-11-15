# Sistema adaptativo con IA para el autocuidado y el turismo rural (ANUC Villeta)

Código del Trabajo de Fin de Máster «Sistema adaptativo con inteligencia artificial para el
autocuidado y el turismo rural en la ANUC Villeta» (UNIR, Máster Universitario en Inteligencia
Artificial). Incluye el motor adaptativo (BKT y bandidos contextuales con máscara de seguridad), el
generador de perfiles sintéticos y el simulador, el recomendador de rutas, el asistente RAG, el
servicio de voz y la API.

> **Estado de este repositorio.** El motor adaptativo, el simulador, el generador, el recomendador de
> rutas, el análisis estadístico y el benchmark están implementados y probados (`pytest`: 66 pruebas).
> El asistente RAG y la voz se probaron con los respaldos sin red (`EmbedderHash`, `VerificadorLexico`,
> `GeneradorExtractivo`); los modelos de producción (E5, ChromaDB, NLI, Whisper, VITS, el LLM) y los
> endpoints de FastAPI están escritos pero **no se han ejecutado todavía**. Los respaldos son solo para
> pruebas: no deben usarse para evaluar el asistente.

## Estructura

```
adaptativo/
    config.py         # 14 competencias, 336 acciones, pesos de la recompensa
    bkt.py            # Actualización BKT (Ecuaciones 4.1 a 4.3)
    contexto.py       # Vector de contexto x_t, d = 32
    acciones.py       # Máscara de seguridad sobre las 336 acciones
    recompensa.py     # Recompensa observable y de evaluación (Ecuación 4.4)
    politicas.py      # Siete políticas, entre ellas LinUCB y Thompson lineal
    perfiles.py       # Generador de perfiles sintéticos
    entorno.py        # Simulador de las personas aprendices
rutas/
    optw.py           # Heurística de inserción y búsqueda local; óptimo exacto
    evaluar_optw.py   # Comparación en 30 instancias
api/  rag/  voz/      # Orquestación FastAPI, asistente RAG y servicio de voz
experimentos/
    ejecutar.py       # Escenarios E1-E6, políticas y semillas 0-49
    analizar.py       # Bootstrap, Wilcoxon, Holm y delta de Cliff
    capturas.py       # Capturas del DataFrame y resumen estadístico (Sección 4.3)
    diagnostico.py    # Conducta de cada política (actividades sobre competencias ya dominadas)
    registros/        # Resultados de la ejecución completa (CSV), tablas y figuras
tests/                # Pruebas unitarias (pytest)
bench.py              # Coste por decisión del motor adaptativo
```

## Instalación y pruebas

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m pytest
```

## Reproducir el experimento

```
python -m experimentos.ejecutar --semillas 0-9 --escenarios E1      # ejecución piloto
python -m experimentos.ejecutar --workers 2                          # 2100 ejecuciones (42 000 000 decisiones)
python -m experimentos.ejecutar --sensibilidad                       # sensibilidad de alpha_ucb y v
python -m experimentos.analizar --validar-generador                  # tablas, H1, figura y validez del generador
python -m experimentos.diagnostico                                   # conducta de cada política (5 semillas, E1)
python -m experimentos.capturas                                      # capturas de la Sección 4.3
python -m rutas.evaluar_optw                                         # heurística frente al óptimo (30 instancias)
python bench.py                                                      # coste por decisión
```

La ejecución es reanudable: las combinaciones ya registradas en el CSV se omiten. Cada fila guarda la
huella SHA-256 de la población, la versión del código (hash de git) y la versión de la configuración.
En una máquina de 2 núcleos, el experimento completo tarda unos 26 minutos.

La carpeta `experimentos/registros/` incluye los resultados de la ejecución completa que respaldan las Tablas 4.16 a 4.20 de la memoria (todos los datos son sintéticos). Los tiempos de `bench.txt` dependen de la máquina.

## Supuestos del simulador que la memoria no fija

El texto de la memoria describe el simulador sin fijar todos los detalles; estas decisiones son del
código y se pueden cambiar en `adaptativo/config.py` y `adaptativo/entorno.py`:

* Cada interacción es una sesión; entre sesiones pasan 1, 2 o 3 días.
* Solo la dificultad modifica la dinámica (actividad avanzada sobre competencia no dominada: la mitad
  de probabilidad de aprendizaje y el doble de abandono); el tipo y el formato no la modifican.
* Una sesión cortada por conectividad cuenta como abandono y no produce respuesta ni aprendizaje.
* La estimación inicial del sistema es p0 (diagnóstico inicial); en E2 es 0,25.
* Las políticas de referencia (secuencia fija y reglas expertas) fijan tipo y formato con reglas
  sencillas descritas en `adaptativo/politicas.py`.
* Las condiciones diabetes, osteomuscular y embarazo usan probabilidades supuestas
  (`adaptativo/perfiles.py`); solo la hipertensión figura en la Tabla 4.8 de la memoria.

## Asistente RAG, voz y API

El corpus son archivos `.md` o `.txt` con la ficha de control de la Tabla 4.7 en la cabecera
(`rag/corpus.py`); solo se indexan los que tienen `estado: aprobado`. **El repositorio no incluye
documentos del corpus** (los derechos de la IFRC están reservados). Las listas de señales de alarma y
de alcance clínico de `rag/filtros.py` son un borrador que debe revisar y firmar el profesional de salud.
Para el prototipo completo: `pip install -r requirements-ia.txt`, `MODO_RAG=produccion`, `VOZ=1` y un
servicio de LLM compatible con la API de chat de OpenAI (variables `LLM_URL`, `LLM_MODELO`). Después:
`API_KEY=... RAG_CORPUS=corpus/ uvicorn api.main:app`.

## Cita y licencia

Licencia: MIT (archivo `LICENSE`). Para citar el software, use `CITATION.cff`; el DOI de Zenodo se añadirá aquí al publicar la versión: [pendiente].
