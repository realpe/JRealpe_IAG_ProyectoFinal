#!/usr/bin/env python3
"""
herramientas.py — Las herramientas (tools) del agente de devoluciones de EcoMarket.

Cuatro herramientas, todas deterministas salvo la última, que envuelve el RAG del Taller 2:

    consultar_pedido(id_pedido)                       -> datos del pedido y sus productos
    verificar_elegibilidad(id_pedido, producto, motivo, empaque_abierto)
                                                      -> decisión por política + código de autorización
    generar_etiqueta_devolucion(id_pedido, sku, autorizacion)
                                                      -> guía de transporte (solo con autorización válida)
    buscar_politicas(pregunta)                        -> fragmentos relevantes del corpus (RAG)

Decisiones de diseño:
  - El LLM nunca transporta identificadores que no puede conocer: `producto` acepta el nombre o
    el SKU y la herramienta lo resuelve contra el pedido. Si no resuelve, devuelve un error con
    los productos del pedido, para que el agente se corrija en la siguiente iteración.
  - La etiqueta exige un código de autorización que solo emite verificar_elegibilidad cuando la
    decisión es ELEGIBLE. Es un HMAC sobre (pedido, sku, decisión): el modelo no puede generar
    etiquetas por pedido del usuario sin pasar por la verificación, aunque se lo ordenen.
  - Cada llamada se registra (REGISTRO) con sus argumentos y su resultado: es el audit trail
    que la Fase 3 propone como monitoreo.
  - Los datos son simulados (datos/pedidos.json). Las etiquetas generadas se guardan en memoria
    para poder consultarlas y para detectar duplicados.
"""
import hashlib
import hmac
import json
import os
import re
import secrets
from datetime import date, datetime
from pathlib import Path

from langchain_core.tools import tool

import reglas_devolucion as reglas

RAIZ = Path(__file__).parent
# Clave del HMAC de autorización. En producción vendría de un secreto gestionado; aquí, del
# entorno o una aleatoria por proceso (basta para impedir que el LLM la adivine).
_CLAVE = os.environ.get("ECOMARKET_CLAVE_AUTORIZACION", secrets.token_hex(16)).encode()

PEDIDOS = reglas.cargar_pedidos(RAIZ / "datos" / "pedidos.json")
ETIQUETAS = {}      # (id_pedido, sku) -> etiqueta emitida
AUTORIZADAS = {}    # id_pedido -> {sku: precio} ya autorizados: evita fraccionar un reclamo grande en varios pequeños
REGISTRO = []       # audit trail en memoria: una entrada por llamada a herramienta
MAX_ETIQUETAS_POR_PEDIDO = 2   # más etiquetas para un mismo pedido requieren revisión humana


def reiniciar():
    """Borra el estado simulado (etiquetas y autorizaciones). Lo usa el evaluador entre casos."""
    ETIQUETAS.clear()
    AUTORIZADAS.clear()


def _registrar(herramienta, argumentos, resultado):
    REGISTRO.append({"ts": datetime.now().isoformat(timespec="seconds"), "herramienta": herramienta,
                     "argumentos": argumentos, "resultado": resultado})
    return resultado


def _firmar(id_pedido, sku):
    return hmac.new(_CLAVE, f"{id_pedido}|{sku}|ELEGIBLE".encode(), hashlib.sha256).hexdigest()[:16]


_VACIAS = {"de", "la", "el", "del", "un", "una", "x", "mi", "los", "las"}


def _tokens(texto):
    """Palabras en minúscula, con los números separados de sus unidades ('500ml' -> '500', 'ml')."""
    t = re.sub(r"(\d+)([a-záéíóúñ]+)", r"\1 \2", (texto or "").lower())
    return set(re.findall(r"[a-záéíóúñ0-9]+", t)) - _VACIAS


def _resolver_item(pedido, producto):
    """Devuelve (item, candidatos). SKU exacto gana; si no, el ítem cuyo nombre comparte más
    palabras con lo que dijo el cliente. Con empate entre varios, no se adivina: item=None y
    candidatos lista los empatados, para que el agente pregunte al cliente."""
    texto = (producto or "").strip().lower()
    for it in pedido["items"]:
        if it["sku"].lower() == texto:
            return it, []
    palabras = _tokens(texto)
    puntajes = [(len(palabras & _tokens(it["nombre"])), it) for it in pedido["items"]]
    maximo = max((p for p, _ in puntajes), default=0)
    if maximo == 0:
        return None, []
    empatados = [it for p, it in puntajes if p == maximo]
    return (empatados[0], []) if len(empatados) == 1 else (None, empatados)


def _pedido(id_pedido):
    return PEDIDOS.get((id_pedido or "").strip().upper())


# ------------------------------------------------------------------ herramientas
@tool
def consultar_pedido(id_pedido: str) -> dict:
    """Devuelve los datos de un pedido de EcoMarket: estado, fecha de entrega y productos (sku,
    nombre, categoría, precio). Úsala siempre que el cliente mencione un número de pedido, y
    antes de verificar la elegibilidad de una devolución. Si el pedido no existe, devuelve
    encontrado=false."""
    p = _pedido(id_pedido)
    if not p:
        return _registrar("consultar_pedido", {"id_pedido": id_pedido},
                          {"encontrado": False, "mensaje": f"No existe el pedido '{id_pedido}'. Pida al cliente "
                           "verificar el número (formato EM-####)."})
    dias = (date.today() - p["fecha_entrega"]).days if p["fecha_entrega"] else None
    nota = None
    if p["estado"] != "ENTREGADO":
        nota = ("El pedido aún no ha sido entregado: no puede devolverse hasta que se entregue, pero el cliente "
                "puede cancelarlo mientras no llegue (FAQ-09).")
    return _registrar("consultar_pedido", {"id_pedido": id_pedido}, {
        "encontrado": True, "id_pedido": p["id"], "estado": p["estado"], "nota": nota,
        "fecha_entrega": p["fecha_entrega"].isoformat() if p["fecha_entrega"] else None,
        "dias_desde_entrega": dias,
        "items": [{"sku": i["sku"], "nombre": i["nombre"], "categoria": i["categoria"], "precio_cop": i["precio_cop"]}
                  for i in p["items"]]})


@tool
def verificar_elegibilidad(id_pedido: str, producto: str, motivo: str = "otro", empaque_abierto: bool = False,
                           devolver_pedido_completo: bool = False) -> dict:
    """Verifica si un producto de un pedido puede devolverse según la política de EcoMarket.
    Llama antes a consultar_pedido. `producto` es el SKU o el nombre del producto tal como
    aparece en el pedido. `motivo` es uno de: no_me_gusto (cambió de opinión), llego_danado
    (roto, abollado, empaque roto), defectuoso (no funciona, vencido, en mal estado, contaminado
    o con cuerpos extraños como pelos o insectos), incompleto (faltan piezas o unidades),
    reaccion_salud, otro; si el cliente no lo dijo, usa 'otro' y verifica sin preguntar.
    `empaque_abierto` indica si el cliente ya abrió el empaque (importa en higiene personal).
    `devolver_pedido_completo` = true cuando el cliente quiere devolver todo el pedido: el
    reclamo se evalúa por el valor total. Devuelve la decisión (ELEGIBLE, NO_ELEGIBLE o
    ESCALAR), la sección de la política que la sustenta y, si es elegible, el código de
    autorización interno que exige generar_etiqueta_devolucion (no se muestra al cliente)."""
    args = {"id_pedido": id_pedido, "producto": producto, "motivo": motivo, "empaque_abierto": empaque_abierto,
            "devolver_pedido_completo": devolver_pedido_completo}
    p = _pedido(id_pedido)
    if not p:
        return _registrar("verificar_elegibilidad", args,
                          {"error": True, "mensaje": f"No existe el pedido '{id_pedido}'."})
    item, candidatos = _resolver_item(p, producto)
    if item is None and candidatos:
        return _registrar("verificar_elegibilidad", args, {
            "error": True, "mensaje": f"'{producto}' es ambiguo en el pedido {p['id']}; pregunte al cliente cuál.",
            "candidatos": [f"{i['sku']} ({i['nombre']})" for i in candidatos]})
    if item is None:
        return _registrar("verificar_elegibilidad", args, {
            "error": True, "mensaje": f"'{producto}' no está en el pedido {p['id']}.",
            "productos_del_pedido": [f"{i['sku']} ({i['nombre']})" for i in p["items"]]})
    if motivo not in reglas.MOTIVOS:
        return _registrar("verificar_elegibilidad", args, {
            "error": True, "mensaje": f"Motivo '{motivo}' no válido.", "motivos_validos": list(reglas.MOTIVOS)})
    # Valor del reclamo: el ítem más lo ya autorizado en el mismo pedido (un reclamo grande no se
    # puede fraccionar en varios pequeños), o el pedido completo si el cliente devuelve todo.
    previas = AUTORIZADAS.get(p["id"], {})
    if devolver_pedido_completo:
        valor = sum(i["precio_cop"] for i in p["items"])
    else:
        valor = item["precio_cop"] + sum(v for k, v in previas.items() if k != item["sku"])
    d = reglas.evaluar(p, item, motivo, empaque_abierto=bool(empaque_abierto), valor_reclamo=valor)
    salida = {"error": False, "id_pedido": p["id"], "sku": item["sku"], "producto": item["nombre"],
              "valor_reclamo_cop": valor, **d}
    if d["decision"] == "ELEGIBLE":
        AUTORIZADAS.setdefault(p["id"], {})[item["sku"]] = item["precio_cop"]
        salida["autorizacion"] = _firmar(p["id"], item["sku"])
        salida["siguiente_paso"] = ("Llame ahora a generar_etiqueta_devolucion con este id_pedido, este sku y esta "
                                    "autorizacion; solo esa herramienta emite el número de guía. No muestre la "
                                    "autorización al cliente.")
    return _registrar("verificar_elegibilidad", args, salida)


@tool
def generar_etiqueta_devolucion(id_pedido: str, sku: str, autorizacion: str) -> dict:
    """Genera la etiqueta de transporte para devolver un producto. Requiere el código de
    `autorizacion` que devolvió verificar_elegibilidad para ese pedido y SKU; sin él, o si el
    producto no era elegible, la herramienta se niega. Devuelve el número de guía, la
    transportadora y las instrucciones para el cliente."""
    args = {"id_pedido": id_pedido, "sku": sku, "autorizacion": autorizacion}
    p = _pedido(id_pedido)
    if not p:
        return _registrar("generar_etiqueta_devolucion", args, {"error": True, "mensaje": f"No existe el pedido '{id_pedido}'."})
    item = next((i for i in p["items"] if i["sku"].lower() == (sku or "").strip().lower()), None)
    if item is None:
        return _registrar("generar_etiqueta_devolucion", args,
                          {"error": True, "mensaje": f"El SKU '{sku}' no está en el pedido {p['id']}."})
    if not hmac.compare_digest(str(autorizacion or ""), _firmar(p["id"], item["sku"])):
        return _registrar("generar_etiqueta_devolucion", args, {
            "error": True, "mensaje": "Autorización inválida. La etiqueta solo se emite después de que "
            "verificar_elegibilidad devuelva ELEGIBLE para este pedido y producto."})
    clave = (p["id"], item["sku"])
    if clave in ETIQUETAS:
        return _registrar("generar_etiqueta_devolucion", args, {**ETIQUETAS[clave], "aviso": "Ya existía una etiqueta; se devuelve la misma."})
    emitidas = sum(1 for (pid, _) in ETIQUETAS if pid == p["id"])
    if emitidas >= MAX_ETIQUETAS_POR_PEDIDO:
        return _registrar("generar_etiqueta_devolucion", args, {
            "error": True, "decision": "ESCALAR", "seccion_politica": "1.4",
            "mensaje": f"Ya se emitieron {emitidas} etiquetas para el pedido {p['id']}. Devolver más productos del "
                       "mismo pedido requiere revisión de un agente humano; informe al cliente que se escala el caso."})
    guia = "ECO-" + hashlib.sha256(f"{p['id']}{item['sku']}".encode()).hexdigest()[:10].upper()
    etiqueta = {"error": False, "numero_guia": guia, "transportadora": "Servientrega (simulada)",
                "id_pedido": p["id"], "sku": item["sku"], "producto": item["nombre"],
                "vigencia_dias": 7, "instrucciones": "Imprima la etiqueta, empaque el producto con sus "
                "accesorios y entréguelo en cualquier punto de la transportadora dentro de los 7 días. "
                "El reembolso se acredita 5 a 10 días hábiles después de recibido."}
    ETIQUETAS[clave] = etiqueta
    return _registrar("generar_etiqueta_devolucion", args, etiqueta)


_recuperador = None


@tool
def buscar_politicas(pregunta: str) -> dict:
    """Busca en las políticas, preguntas frecuentes, catálogo y guía de cuidado de EcoMarket los
    fragmentos que responden una pregunta general (plazos, envíos, garantías, cuidado de
    productos). No sirve para consultar un pedido concreto. Devuelve los fragmentos con su id,
    para citarlos."""
    global _recuperador
    if _recuperador is None:
        import rag_ecomarket
        _recuperador = rag_ecomarket.crear_recuperador()
    docs = _recuperador.recuperar(pregunta)
    return _registrar("buscar_politicas", {"pregunta": pregunta},
                      {"fragmentos": [{"id": d.id, "texto": d.page_content} for d in docs]})


HERRAMIENTAS = [consultar_pedido, verificar_elegibilidad, generar_etiqueta_devolucion, buscar_politicas]
POR_NOMBRE = {t.name: t for t in HERRAMIENTAS}
