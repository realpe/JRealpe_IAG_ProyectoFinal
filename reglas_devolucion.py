#!/usr/bin/env python3
"""
reglas_devolucion.py — La política de devoluciones de EcoMarket como código determinista.

Implementa las secciones 1.1 a 1.4 de corpus/politicas.md tal como están escritas. Cada
decisión devuelve la sección que la sustenta, para que el agente la cite y para que un auditor
pueda comprobar la regla contra el documento. Ningún modelo de lenguaje interviene aquí: las
decisiones con consecuencias (dinero, salud, escalamiento) no se delegan al LLM.

Decisiones posibles:
    ELEGIBLE      procede reposición o reembolso; se puede generar la etiqueta
    NO_ELEGIBLE   la política lo excluye; se explica al cliente con la sección
    ESCALAR       un agente humano debe decidir (sección 1.4)

Uso del módulo: evaluar(pedido, item, motivo, empaque_abierto, hoy) -> dict
"""
from datetime import date, timedelta

PLAZO_DIAS = 30                 # 1.1
HORAS_REPORTE_DANO = 72         # 1.3
TOPE_RECLAMO_COP = 500_000      # 1.4
VENTANA_SEGUNDA_DEVOLUCION = 60 # 1.4

MOTIVOS = ("no_me_gusto", "llego_danado", "defectuoso", "incompleto", "reaccion_salud", "otro")
MOTIVOS_DANO = ("llego_danado", "defectuoso", "incompleto")   # activan la excepción 1.3


def _decision(decision, seccion, explicacion, **extra):
    return {"decision": decision, "seccion_politica": seccion, "explicacion": explicacion, **extra}


def evaluar(pedido, item, motivo, empaque_abierto=False, hoy=None, valor_reclamo=None):
    """pedido: dict con estado, fecha_entrega (date), devoluciones_previas_dias (list[int]).
    item: dict con sku, nombre, categoria, precio_cop. valor_reclamo: COP del reclamo total
    (por defecto, el precio del ítem)."""
    hoy = hoy or date.today()
    if motivo not in MOTIVOS:
        raise ValueError(f"motivo inválido: {motivo}; use uno de {MOTIVOS}")
    valor = valor_reclamo if valor_reclamo is not None else item["precio_cop"]

    # --- 1.4 casos que el asistente no puede resolver: se evalúan primero, porque priman
    if motivo == "reaccion_salud":
        return _decision("ESCALAR", "1.4", "Cualquier mención de daño a la salud, reacción alérgica o "
                         "intoxicación debe atenderla un agente humano.", requiere_humano=True)
    if valor > TOPE_RECLAMO_COP:
        return _decision("ESCALAR", "1.4", f"Los reclamos por más de ${TOPE_RECLAMO_COP:,} COP requieren "
                         "aprobación de un agente humano.".replace(",", "."), requiere_humano=True)
    if any(d <= VENTANA_SEGUNDA_DEVOLUCION for d in pedido.get("devoluciones_previas_dias", [])):
        return _decision("ESCALAR", "1.4", "Es la segunda devolución del mismo cliente en un período de "
                         f"{VENTANA_SEGUNDA_DEVOLUCION} días; la revisa un agente humano.", requiere_humano=True)

    # --- estado del pedido
    if pedido["estado"] != "ENTREGADO":
        return _decision("NO_ELEGIBLE", "FAQ-09", "El pedido aún no ha sido entregado, así que no puede "
                         "devolverse. Mientras no se entregue, puede cancelarse.", accion_alterna="cancelar")

    dias = (hoy - pedido["fecha_entrega"]).days
    horas = dias * 24

    # --- 1.3 excepción por producto dañado, incompleto o defectuoso: prima sobre 1.2
    if motivo in MOTIVOS_DANO:
        if horas <= HORAS_REPORTE_DANO:
            return _decision("ELEGIBLE", "1.3", "El producto llegó dañado, incompleto o defectuoso. Aplica la "
                             "excepción: cualquier categoría da derecho a reposición o reembolso total, sin "
                             "importar el estado del empaque. Requiere evidencia fotográfica; EcoMarket asume "
                             "el transporte de retorno.", requiere_evidencia=True, transporte="EcoMarket")
        # fuera de las 72 horas la excepción no aplica: se sigue con la regla general de la categoría
        nota = (f"El daño se reporta {dias} días después de la entrega, fuera de las "
                f"{HORAS_REPORTE_DANO} horas de la sección 1.3, así que aplica la regla general. ")
    else:
        nota = ""

    # --- 1.1 plazo general
    if dias > PLAZO_DIAS:
        return _decision("ESCALAR", "1.4", nota + f"Han pasado {dias} días desde la entrega; el plazo es de "
                         f"{PLAZO_DIAS} días calendario (1.1). Las solicitudes fuera de plazo requieren "
                         "aprobación discrecional de un agente humano.", requiere_humano=True)

    # --- 1.2 elegibilidad por categoría
    cat = item["categoria"]
    if cat == "PERECEDERO":
        return _decision("NO_ELEGIBLE", "1.2", nota + "Los alimentos y productos con fecha de vencimiento "
                         "no se reciben de vuelta.")
    if cat == "HIGIENE_PERSONAL":
        if empaque_abierto:
            return _decision("NO_ELEGIBLE", "1.2", nota + "Por normativa sanitaria, un producto de higiene con "
                             "el sello roto no puede reingresar al inventario.")
        return _decision("ELEGIBLE", "1.2", nota + "Producto de higiene personal con el empaque sin abrir: "
                         "admite devolución.", transporte="cliente")
    if cat == "DURADERO":
        return _decision("ELEGIBLE", "1.2", nota + "Producto duradero dentro del plazo: admite devolución si "
                         "está sin uso, con empaque original y etiquetas. Reembolso al medio de pago original "
                         "en 5 a 10 días hábiles.", transporte="cliente")
    return _decision("ESCALAR", "4", f"Categoría '{cat}' no contemplada en la política.", requiere_humano=True)


def cargar_pedidos(ruta, hoy=None):
    """Lee datos/pedidos.json y convierte 'dias_desde_entrega' en 'fecha_entrega' (date)."""
    import json
    hoy = hoy or date.today()
    datos = json.loads(open(ruta, encoding="utf-8").read())["pedidos"]
    salida = {}
    for p in datos:
        q = dict(p)
        q["fecha_entrega"] = (hoy - timedelta(days=p["dias_desde_entrega"])) if p["dias_desde_entrega"] is not None else None
        salida[p["id"]] = q
    return salida
