#!/usr/bin/env python3
"""
router.py — Decide por reglas a qué ruta va cada mensaje, antes de que intervenga un LLM.

Rutas:
    riesgo       salud, reacción alérgica, intoxicación, amenaza legal o de exposición pública:
                 se escala a un agente humano de inmediato (sección 1.4 de las políticas). El LLM
                 no interviene, de modo que ningún prompt puede convencerlo de otra cosa.
    accion       el cliente quiere hacer algo con un pedido (devolver, cambiar, reembolso,
                 etiqueta, cancelar, estado de un pedido): va al agente con herramientas.
    informativa  todo lo demás: pregunta general que responde el RAG del Taller 2.

Es determinista a propósito. Un clasificador con LLM sería más flexible, pero aquí la ruta de
riesgo es una decisión de seguridad y debe ser auditable con una tabla de casos (test_router).
Límite conocido: las reglas dependen del vocabulario; una queja de salud dicha con palabras no
previstas llegaría al agente, que de todos modos escala por `motivo=reaccion_salud` si lo detecta.
"""
import re
import unicodedata

PATRON_PEDIDO = re.compile(r"\bEM-?\s?(\d{4})\b", re.I)

_RIESGO = [r"\bsarpullid", r"\balergi", r"\balergic", r"\bintoxic", r"\breaccion\b", r"\bronch", r"\bpicazon",
           r"\bhospital", r"\bmedic[oa]\b", r"\burgencias", r"\bdemand", r"\babogad", r"\bsuperintendencia",
           r"\bsic\b", r"\bredes sociales", r"\btwitter|\binstagram|\btiktok", r"\bdenunci", r"\bacci[oó]n legal"]
# Una acción exige intención en primera persona sobre un pedido o producto. Una pregunta general
# ("¿cuántos días tengo para devolver?", "¿se pueden devolver los alimentos?") es informativa y la
# responde el RAG, aunque contenga la palabra "devolver".
_VERBOS_ACCION = r"(devol\w*|devuelv\w*|reembols\w*|cambi(ar|o)\b|cancel\w*|etiqueta|guia\b)"
_ACCION = [r"\b(quiero|necesito|deseo|quisiera|me gustaria|voy a|puedo|podria|podrian)\b[^.?!]{0,40}" + _VERBOS_ACCION,
           r"\b(me|nos) (llego|llegaron|entregaron)\b", r"\bmi pedido\b", r"\bestado de(l| mi)? pedido", r"\brastre",
           r"\b(roto|rota|danad\w*|abollad\w*|partid\w*|quebrad\w*|defectuos\w*|incomplet\w*)\b",
           r"\bgenera\w*\b[^.?!]{0,30}\betiqueta"]


def _normalizar(texto):
    t = unicodedata.normalize("NFKD", texto or "").encode("ascii", "ignore").decode().lower()
    return re.sub(r"\s+", " ", t)


def enrutar(mensaje):
    """Devuelve {'ruta': ..., 'pedidos': [...], 'motivo': ...}."""
    t = _normalizar(mensaje)
    pedidos = ["EM-" + m for m in PATRON_PEDIDO.findall(mensaje or "")]
    for p in _RIESGO:
        if re.search(p, t):
            return {"ruta": "riesgo", "pedidos": pedidos, "motivo": f"coincide '{p}'"}
    if pedidos:
        return {"ruta": "accion", "pedidos": pedidos, "motivo": "menciona un número de pedido"}
    for p in _ACCION:
        if re.search(p, t):
            return {"ruta": "accion", "pedidos": pedidos, "motivo": f"coincide '{p}'"}
    return {"ruta": "informativa", "pedidos": pedidos, "motivo": "sin señales de acción ni riesgo"}
