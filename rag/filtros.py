"""Filtros previos del asistente RAG: señales de alarma y alcance clínico (Subsección 4.2.7).

Se ejecutan FUERA del modelo de lenguaje y no pueden desactivarse desde la consulta. Las listas
de este archivo son un borrador técnico: antes de cualquier uso con personas deben ser revisadas
y firmadas por el profesional de salud que valida el corpus (Subsección 4.2.10).
"""
import re

from .embeddings import normalizar

# Descripciones de urgencia vital (se comparan sobre el texto sin tildes y en minúsculas)
PATRONES_ALARMA = [
    r"hemorragia (abundante|fuerte|grave)|sangr\w+ (mucho|sin parar|abundante)|no para de sangrar",
    r"dificultad (para|al) respirar|no (puedo|puede|logra) respirar|se ahoga|ahogando",
    r"perdi\w* (el )?(conocimiento|la conciencia)|desmay\w+|inconsciente|no responde|no reacciona",
    r"cay\w+ de (una )?altura|cayo (de|desde) (un |el )?(arbol|techo|escalera|caballo)",
    r"mordedura de (serpiente|culebra)|mordio (una |la )?(serpiente|culebra)|me pico (una |la )?serpiente",
    r"dolor (fuerte )?(en el pecho|de pecho)|infarto",
    r"convuls\w+|ataque epileptico",
    r"quemadura (extensa|grave|en la cara)|quemo (toda|gran parte)",
    r"intoxicad\w+|envenenad\w+|tomo (veneno|agroquimico|plaguicida)|ingirio (veneno|plaguicida)",
    r"golpe fuerte en la cabeza|traumatismo (craneal|de craneo)",
    r"embarazada .*(sangrado|dolor fuerte)|sangrado .*embarazada",
]

# Peticiones de diagnóstico, de dosis o de tratamiento
PATRONES_CLINICOS = [
    r"\b(que|cual) (enfermedad|tengo|tiene)\b",
    r"\b(diagnostic\w+|diagnostico)\b",
    r"\b(cuanto|cuantas?|que) (dosis|pastillas?|gotas|miligramos|mg)\b",
    r"\bdosis\b",
    r"\b(que|cual) (me|le) (recomienda|receta|doy|puedo dar|puedo tomar)\b",
    r"\b(puedo|debo|podria) (tomar|darle|darme|inyectar\w*|recetar\w*)\b.*\b(medicamento|pastilla|antibiotico|ibuprofeno|acetaminofen|aspirina|insulina)\b",
    r"\b(tratamiento|tratar|curar) (para|de|la|el)\b.*\b(enfermedad|infeccion|cancer|diabetes|hipertension)\b",
]

MENSAJE_EMERGENCIA = (
    "Esto puede ser una emergencia. Llame ya a la línea de emergencias {linea} o pida a alguien "
    "cercano que la llame, y busque atención médica de inmediato. No espere a que pase. "
    "Este asistente no puede reemplazar la atención de un profesional de salud."
)
MENSAJE_DERIVACION = (
    "No puedo dar un diagnóstico, una dosis ni un tratamiento. Para eso hable con un profesional de "
    "salud o consulte en el puesto de salud más cercano. Si la situación es urgente, llame a la "
    "línea {linea}. Con gusto le explico medidas generales de prevención y primeros auxilios."
)
MENSAJE_ABSTENCION = (
    "No cuento con información respaldada en los documentos aprobados para responder a esa "
    "pregunta. Le recomiendo consultar a un profesional o a la entidad correspondiente."
)


def detectar_alarma(texto):
    """True si la consulta describe una urgencia vital."""
    t = normalizar(texto)
    return any(re.search(p, t) for p in PATRONES_ALARMA)


def pide_alcance_clinico(texto):
    """True si la consulta pide un diagnóstico, una dosis o un tratamiento."""
    t = normalizar(texto)
    return any(re.search(p, t) for p in PATRONES_CLINICOS)


def delimitar_datos(fragmentos):
    """Inserta los fragmentos delimitados y marcados como datos, no como órdenes.

    Es una de las tres defensas frente a la inyección de instrucciones (Perez y Ribeiro, 2022;
    Greshake et al., 2023). Se neutralizan además las etiquetas de cierre que pudieran venir
    en el texto de un documento.
    """
    bloques = []
    for i, f in enumerate(fragmentos, 1):
        limpio = f.texto.replace("<DATOS", "< DATOS").replace("</DATOS", "< /DATOS")
        bloques.append(f"<DATOS fuente=[{i}] documento=\"{f.documento}\" seccion=\"{f.seccion}\">\n"
                       f"{limpio}\n</DATOS>")
    return "\n\n".join(bloques)
