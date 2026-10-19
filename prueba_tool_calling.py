#!/usr/bin/env python3
"""
prueba_tool_calling.py — ¿El modelo decide bien cuándo llamar una herramienta y con qué argumentos?

Es la prueba previa al diseño del agente del Proyecto Final. Un agente que no llama la
herramienta correcta con argumentos válidos no sirve, por bueno que sea redactando. Se mide:

  decisión   llama una herramienta cuando el prompt lo requiere, y NO la llama cuando es una
             pregunta informativa o un saludo
  elección   la herramienta llamada es la esperada
  argumentos los argumentos son válidos (el id de pedido existe en el prompt; el SKU también)
  cierre     en un caso completo de dos pasos (llamada -> resultado -> respuesta), el modelo
             produce una respuesta final en lenguaje natural que usa el resultado

Cada prompt se repite N veces (--repeticiones), porque con modelos locales la decisión puede
variar entre corridas aunque la temperatura sea 0 cuando cambia el orden de las herramientas
o el historial; aquí el prompt es idéntico, así que la repetición mide estabilidad.

Uso:
    python prueba_tool_calling.py                                   # qwen3:14b local
    python prueba_tool_calling.py --modelo anthropic:claude-haiku-4-5-20251001
    python prueba_tool_calling.py --repeticiones 3
"""
import argparse
import json
import os
import re
import time

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

OLLAMA = "http://localhost:11434"

# ------------------------------------------------------------------ herramientas (simuladas)
PEDIDOS = {
    "EM-1001": {"fecha_entrega": "2026-10-10", "estado": "ENTREGADO",
                "items": [{"sku": "ECO-TER-1L", "nombre": "Termo de acero 1L", "categoria": "DURADERO", "precio_cop": 121000}]},
    "EM-1002": {"fecha_entrega": "2026-10-12", "estado": "ENTREGADO",
                "items": [{"sku": "ECO-FRU-MIX", "nombre": "Mix de frutos secos 1kg", "categoria": "PERECEDERO", "precio_cop": 78000}]},
    "EM-1003": {"fecha_entrega": "2026-08-01", "estado": "ENTREGADO",
                "items": [{"sku": "ECO-MOC-M", "nombre": "Mochila de algodón reciclado", "categoria": "DURADERO", "precio_cop": 96000}]},
}


@tool
def consultar_pedido(id_pedido: str) -> dict:
    """Devuelve los datos de un pedido de EcoMarket: fecha de entrega, estado y productos
    (sku, nombre, categoría y precio). Úsala cuando el cliente mencione un número de pedido."""
    p = PEDIDOS.get(id_pedido.strip().upper())
    return {"encontrado": bool(p), "id_pedido": id_pedido, **(p or {})}


def _resolver_item(pedido, producto):
    """Empareja de forma determinista lo que dijo el cliente con un ítem del pedido: por SKU
    exacto o por coincidencia de palabras del nombre. El LLM nunca tiene que adivinar un código."""
    texto = producto.strip().lower()
    for it in pedido["items"]:
        if it["sku"].lower() == texto:
            return it
    palabras = set(re.findall(r"[a-záéíóúñ0-9]+", texto))
    mejor, puntaje = None, 0
    for it in pedido["items"]:
        comunes = palabras & set(re.findall(r"[a-záéíóúñ0-9]+", it["nombre"].lower()))
        if len(comunes) > puntaje:
            mejor, puntaje = it, len(comunes)
    return mejor


@tool
def verificar_elegibilidad(id_pedido: str, producto: str, motivo: str) -> dict:
    """Verifica si un producto de un pedido puede devolverse según la política de EcoMarket.
    Llama antes a consultar_pedido para confirmar que el pedido existe y ver sus productos.
    `producto` es el SKU o el nombre del producto tal como aparece en el pedido.
    `motivo` es una de: 'no_me_gusto', 'llego_danado', 'defectuoso', 'otro'.
    Si el producto no está en el pedido, devuelve error=True con los productos disponibles."""
    pedido = PEDIDOS.get(id_pedido.strip().upper())
    if not pedido:
        return {"error": True, "mensaje": f"El pedido {id_pedido} no existe. Pida al cliente verificar el número."}
    item = _resolver_item(pedido, producto)
    if item is None:
        return {"error": True, "mensaje": f"'{producto}' no está en el pedido {id_pedido}.",
                "productos_del_pedido": [f"{i['sku']} ({i['nombre']})" for i in pedido["items"]]}
    return {"error": False, "sku": item["sku"], "elegible": True, "seccion_politica": "1.1", "motivo": "simulado"}


HERRAMIENTAS = [consultar_pedido, verificar_elegibilidad]

SISTEMA = ("Eres el asistente de atención al cliente de EcoMarket. Cuando el cliente pida devolver "
           "un producto o pregunte por un pedido concreto, usa las herramientas disponibles para "
           "consultar el pedido primero y luego verificar la elegibilidad antes de responder. Si la pregunta es "
           "general sobre políticas o un saludo, responde directamente sin herramientas.")

# ------------------------------------------------------------------ casos
# esperado: None = no debe llamar herramienta; o (nombre, {arg: valor esperado})
CASOS = [
    {"id": "c01", "prompt": "Hola, quiero devolver el termo del pedido EM-1001, me llegó abollado.",
     "esperado": ("consultar_pedido", {"id_pedido": "EM-1001"}),
     "alterno": ("verificar_elegibilidad", {"id_pedido": "EM-1001"})},
    {"id": "c02", "prompt": "¿Puedo devolver el mix de frutos secos del pedido EM-1002? No me gustó el sabor.",
     "esperado": ("consultar_pedido", {"id_pedido": "EM-1002"}),
     "alterno": ("verificar_elegibilidad", {"id_pedido": "EM-1002"})},
    {"id": "c03", "prompt": "¿Cuál es el estado de mi pedido EM-1003?",
     "esperado": ("consultar_pedido", {"id_pedido": "EM-1003"}), "alterno": None},
    {"id": "c04", "prompt": "¿Cuántos días tengo para devolver un producto?", "esperado": None, "alterno": None},
    {"id": "c05", "prompt": "Buenas tardes, ¿me pueden ayudar?", "esperado": None, "alterno": None},
    {"id": "c06", "prompt": "Quiero devolver una mochila pero no tengo el número de pedido a la mano.",
     "esperado": None, "alterno": None},   # debe pedir el número, no inventar uno
]


def crear_llm(modelo):
    if modelo.startswith("anthropic:"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Falta ANTHROPIC_API_KEY en el entorno")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=modelo.split(":", 1)[1], max_tokens=512)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=modelo, base_url=OLLAMA, temperature=0, reasoning=False, num_ctx=8192)


def evaluar_decision(respuesta, caso):
    """Devuelve dict con decision/eleccion/argumentos (True/False/None) y detalle."""
    llamadas = getattr(respuesta, "tool_calls", None) or []
    esperado = caso["esperado"]
    if esperado is None:
        return {"decision": not llamadas, "eleccion": None, "argumentos": None,
                "detalle": [c["name"] for c in llamadas] or "sin herramienta"}
    if not llamadas:
        return {"decision": False, "eleccion": False, "argumentos": False, "detalle": "no llamó herramienta"}
    c = llamadas[0]
    opciones = [esperado] + ([caso["alterno"]] if caso["alterno"] else [])
    for nombre, args_esp in opciones:
        if c["name"] == nombre:
            ok_args = all(str(c["args"].get(k, "")).strip().upper() == v.upper() for k, v in args_esp.items())
            if ok_args and nombre == "verificar_elegibilidad":
                # válido solo si la herramienta logra resolver el producto con lo que el modelo pasó
                ok_args = not verificar_elegibilidad.invoke(c["args"]).get("error", True)
            return {"decision": True, "eleccion": True, "argumentos": ok_args,
                    "detalle": f"{c['name']}({json.dumps(c['args'], ensure_ascii=False, sort_keys=True)})"}
    return {"decision": True, "eleccion": False, "argumentos": False,
            "detalle": f"{c['name']}({json.dumps(c['args'], ensure_ascii=False, sort_keys=True)})"}


def caso_completo(llm_t):
    """Dos pasos: llamada -> ejecutar -> respuesta final. Devuelve (ok, texto)."""
    msgs = [SystemMessage(SISTEMA), HumanMessage("¿Cuál es el estado de mi pedido EM-1001 y qué productos tiene?")]
    r1 = llm_t.invoke(msgs)
    if not getattr(r1, "tool_calls", None):
        return False, "no llamó herramienta en el primer paso"
    msgs.append(r1)
    for c in r1.tool_calls:
        fn = {t.name: t for t in HERRAMIENTAS}[c["name"]]
        msgs.append(ToolMessage(content=json.dumps(fn.invoke(c["args"]), ensure_ascii=False), tool_call_id=c["id"]))
    r2 = llm_t.invoke(msgs)
    texto = r2.content if isinstance(r2.content, str) else " ".join(
        (b.get("text", "") if isinstance(b, dict) else str(b)) for b in r2.content)
    ok = bool(texto.strip()) and not getattr(r2, "tool_calls", None) and ("ENTREGADO" in texto.upper() or "entregado" in texto.lower()) and "termo" in texto.lower()
    return ok, texto.strip()[:300]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default="qwen3:14b")
    ap.add_argument("--repeticiones", type=int, default=2)
    a = ap.parse_args()
    llm_t = crear_llm(a.modelo).bind_tools(HERRAMIENTAS)

    print(f"Modelo: {a.modelo} | {len(CASOS)} casos x {a.repeticiones} repeticiones\n")
    tot = {"decision": [0, 0], "eleccion": [0, 0], "argumentos": [0, 0]}
    t0 = time.perf_counter()
    for caso in CASOS:
        resultados = []
        for _ in range(a.repeticiones):
            r = llm_t.invoke([SystemMessage(SISTEMA), HumanMessage(caso["prompt"])])
            ev = evaluar_decision(r, caso)
            resultados.append(ev)
            for k in tot:
                if ev[k] is not None:
                    tot[k][1] += 1
                    tot[k][0] += bool(ev[k])
        estable = len({str(e["detalle"]) for e in resultados}) == 1
        marca = "ok " if all(e["decision"] and e["eleccion"] is not False and e["argumentos"] is not False for e in resultados) else "FALLA"
        print(f"  {marca} {caso['id']} {'estable  ' if estable else 'INESTABLE'} {resultados[0]['detalle']}")
        if not estable:
            for e in resultados[1:]:
                print(f"               {e['detalle']}")
    seg = time.perf_counter() - t0
    print(f"\n  decisión correcta   {tot['decision'][0]}/{tot['decision'][1]}")
    print(f"  herramienta correcta {tot['eleccion'][0]}/{tot['eleccion'][1]}")
    print(f"  argumentos válidos  {tot['argumentos'][0]}/{tot['argumentos'][1]}")
    print(f"  segundos por llamada {seg / (len(CASOS) * a.repeticiones):.1f}")

    print("\nCaso completo (dos pasos):")
    ok, texto = caso_completo(llm_t)
    print(f"  {'ok ' if ok else 'FALLA'} {texto}")


if __name__ == "__main__":
    main()
