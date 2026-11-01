#!/usr/bin/env python3
"""
agente.py — Agente de devoluciones de EcoMarket: router + RAG + bucle de herramientas.

Flujo de un mensaje:
    mensaje -> router.enrutar()
        riesgo       -> respuesta fija de escalamiento (sin LLM) + resumen del caso para el humano
        informativa  -> RAG del Taller 2 (rag_ecomarket.crear_rag): respuesta con citas
        accion       -> bucle del agente:
                          1. el LLM, con las herramientas declaradas, decide: llamar una o responder
                          2. se ejecutan las llamadas; cada resultado (o error) vuelve como ToolMessage
                          3. se repite hasta que el LLM responde en texto, o hasta MAX_ITERACIONES
                        Si se agota el límite, se responde con una derivación a humano, nunca con
                        una respuesta a medias.

El bucle está escrito a mano (unas 40 líneas) en lugar de usar `create_agent` de LangChain, para
que cada paso sea visible, se controle el límite de iteraciones y se registre todo. Cada turno
se escribe en registro_agente.jsonl (ruta, llamadas, decisión, etiqueta, iteraciones, segundos):
es el audit trail que la Fase 3 propone como base del monitoreo.

El modelo se elige por configuración: `qwen3:14b` (Ollama, por defecto) o `anthropic:<id>`.

Uso:
    python agente.py "Quiero devolver el termo del pedido EM-1001, me llegó abollado"
    python agente.py --modelo anthropic:claude-haiku-4-5-20251001 "..."
    python agente.py --interactivo
"""
import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

import herramientas as H
import router

RAIZ = Path(__file__).parent
OLLAMA = "http://localhost:11434"
MODELO_POR_DEFECTO = "qwen3:14b"
MAX_ITERACIONES = 6          # cada iteración es una llamada al LLM; 6 cubre consultar -> verificar -> etiqueta con dos correcciones
REGISTRO_JSONL = RAIZ / "registro_agente.jsonl"

PROMPT_AGENTE = """Eres el asistente de atención al cliente de EcoMarket, una tienda de productos sostenibles en Colombia. Atiendes solicitudes sobre pedidos y devoluciones usando herramientas.

Cómo trabajar:
1. Si el cliente menciona un número de pedido, llama consultar_pedido primero. Si no lo menciona y lo necesitas, pídeselo; nunca inventes un número.
2. Para una devolución, llama verificar_elegibilidad con el pedido, el producto (nombre o SKU tal como aparece en el pedido) y el motivo. Si el cliente dice que el producto llegó dañado, roto, partido, abollado, con el empaque roto, incompleto, vencido, en mal estado, contaminado o con cuerpos extraños (pelos, insectos, moho), el motivo es llego_danado, defectuoso o incompleto, porque la excepción de la sección 1.3 cubre cualquier producto dañado, incompleto o defectuoso; si no dijo motivo, usa 'otro' y verifica de todos modos: nunca preguntes el motivo antes de verificar, porque la verificación puede escalar o rechazar sin importar el motivo. Si el cliente dijo que el producto sigue sellado pasa empaque_abierto=false; si dijo que lo abrió, true; si no lo dijo, no pases ese parámetro: para productos de higiene la herramienta te devolverá la pregunta que debes hacerle al cliente, y en el siguiente turno verificas de nuevo con su respuesta. Si el cliente quiere devolver todo el pedido o 'los artículos' sin nombrar uno, verifica con producto='todo el pedido' y devolver_pedido_completo=true.
3. Si la decisión es ELEGIBLE, llama generar_etiqueta_devolucion con el código de autorización que devolvió la verificación (si la verificación trae una lista `resultados`, una etiqueta por cada producto ELEGIBLE, con la autorización de cada uno), y solo entonces entrega al cliente el número de guía y las instrucciones. El número de guía existe únicamente si lo devolvió esa herramienta: nunca inventes una guía, una transportadora ni un plazo. El código de autorización es interno: no lo muestres al cliente. Si la verificación exige evidencia fotográfica, indícale que debe enviarla.
4. Si la decisión es NO_ELEGIBLE, explica el motivo con la sección de la política y ofrece alternativas si las hay.
5. Si la decisión es ESCALAR, informa que un agente humano revisará el caso y resume lo que ya se sabe.
6. Si una herramienta devuelve error, lee el mensaje: corrige la llamada (por ejemplo, con los productos que lista) o pide al cliente el dato que falta.
7. Para preguntas generales sobre políticas usa buscar_politicas y cita el id del fragmento entre corchetes.

Reglas de comunicación: trata al cliente de usted, sé claro y cordial, máximo 150 palabras, cita siempre la sección de política que sustenta la decisión con la palabra "sección" (por ejemplo "sección 1.3", tomada del campo seccion_politica), nunca prometas fechas exactas ni ofrezcas descuentos o compensaciones, y no pidas ni repitas datos sensibles (documento, tarjeta, dirección completa). Cierra ofreciendo ayuda adicional."""

RESPUESTA_RIESGO = ("Lamento mucho lo que me cuenta. Por tratarse de un tema que involucra su salud o un reclamo "
                    "formal, lo atenderá directamente un agente humano de EcoMarket, que ya recibe el resumen de "
                    "su caso{pedido}. Le pido no usar el producto mientras tanto. ¿Hay algo más que quiera dejar "
                    "anotado para el agente?")


def crear_llm(modelo=MODELO_POR_DEFECTO):
    if modelo.startswith("anthropic:"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("Falta ANTHROPIC_API_KEY en el entorno")
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=modelo.split(":", 1)[1], max_tokens=700)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=modelo, base_url=OLLAMA, temperature=0, reasoning=False, num_ctx=8192)


def _texto(msg):
    c = msg.content
    if isinstance(c, str):
        return c
    return " ".join((b.get("text", "") if isinstance(b, dict) else str(b)) for b in c).strip()


class Agente:
    def __init__(self, modelo=MODELO_POR_DEFECTO, llm=None, rag=None, registrar=True):
        self.modelo = modelo
        self.llm = (llm or crear_llm(modelo)).bind_tools(H.HERRAMIENTAS)
        self._rag = rag              # se crea perezosamente: carga Chroma solo si hace falta
        self.registrar = registrar

    # ---------------------------------------------------------------- rutas
    def _rag_cadena(self):
        if self._rag is None:
            import rag_ecomarket
            self._rag = rag_ecomarket.crear_rag(modelo_llm=self.modelo if not self.modelo.startswith("anthropic:") else rag_ecomarket.MODELO_LLM)
        return self._rag

    def _ruta_riesgo(self, mensaje, info):
        pedido = f" (pedido {info['pedidos'][0]})" if info["pedidos"] else ""
        resumen = {"tipo": "escalamiento_por_riesgo", "mensaje_cliente": mensaje, "pedidos": info["pedidos"],
                   "motivo": info["motivo"], "ts": datetime.now().isoformat(timespec="seconds")}
        return RESPUESTA_RIESGO.format(pedido=pedido), {"escalado": True, "resumen_para_humano": resumen}

    def _ruta_informativa(self, mensaje):
        r = self._rag_cadena().invoke(mensaje)
        return r["respuesta"].strip(), {"fragmentos": [d.id for d in r["docs"]]}

    def _ruta_accion(self, mensaje, historial):
        msgs = [SystemMessage(PROMPT_AGENTE)] + list(historial) + [HumanMessage(mensaje)]
        llamadas, iteraciones, empujado = [], 0, False
        while iteraciones < MAX_ITERACIONES:
            iteraciones += 1
            respuesta = self.llm.invoke(msgs)
            msgs.append(respuesta)
            if not getattr(respuesta, "tool_calls", None):
                # Guardarraíl: si la verificación fue ELEGIBLE y no se generó la etiqueta, el modelo no
                # debe cerrar el turno (es donde qwen3 inventó una guía en la primera evaluación).
                # Se le recuerda una sola vez; si insiste en responder sin etiqueta, se acepta.
                if _ultima_decision(llamadas) == "ELEGIBLE" and not _etiqueta(llamadas) and not empujado:
                    empujado = True
                    msgs.append(HumanMessage("[Sistema] La verificación devolvió ELEGIBLE y aún no se ha generado la "
                                             "etiqueta. Llama a generar_etiqueta_devolucion con la autorización "
                                             "recibida antes de responder al cliente."))
                    continue
                texto = _texto(respuesta)
                # Trazabilidad que no depende del modelo: si hubo una decisión de política y la
                # respuesta no cita la sección, se añade al final.
                seccion = _ultima_seccion(llamadas)
                if seccion and "secci" not in texto.lower():
                    texto = texto.rstrip() + f"\n\n(Sección {seccion} de la política de devoluciones de EcoMarket.)"
                # Igual con el requisito de evidencia de la excepción 1.3: si el modelo lo omite, se añade.
                if _requiere_evidencia(llamadas) and not re.search(r"foto|evidencia", texto.lower()):
                    texto = texto.rstrip() + ("\n\nRecuerde enviar fotografías del producto y del empaque: son el "
                                              "requisito de la sección 1.3 para la devolución por producto dañado.")
                return texto, {"llamadas": llamadas, "iteraciones": iteraciones, "agotado": False, "empujado": empujado}
            for c in respuesta.tool_calls:
                fn = H.POR_NOMBRE.get(c["name"])
                try:
                    resultado = fn.invoke(c["args"]) if fn else {"error": True, "mensaje": f"La herramienta '{c['name']}' no existe."}
                except Exception as e:  # noqa: BLE001 - un fallo de herramienta vuelve al modelo como error, no tumba el turno
                    resultado = {"error": True, "mensaje": f"La herramienta falló: {type(e).__name__}: {e}"}
                llamadas.append({"herramienta": c["name"], "argumentos": c["args"], "resultado": resultado})
                msgs.append(ToolMessage(content=json.dumps(resultado, ensure_ascii=False, default=str), tool_call_id=c["id"]))
        # Límite agotado: nunca se entrega una respuesta a medias
        texto = ("No logré completar su solicitud de forma automática. La paso a un agente humano con todo lo "
                 "que ya se revisó, para que la termine con usted. Disculpe la molestia.")
        return texto, {"llamadas": llamadas, "iteraciones": iteraciones, "agotado": True}

    # ---------------------------------------------------------------- entrada única
    def responder(self, mensaje, historial=(), ruta_previa=None):
        """`ruta_previa` es la ruta del turno anterior de la misma conversación: un mensaje sin
        señales propias ("pero no lo he destapado") continúa la conversación de acción en curso
        en lugar de irse al RAG. El riesgo conserva la prioridad en cualquier caso."""
        t0 = time.perf_counter()
        info = router.enrutar(mensaje)
        if info["ruta"] == "informativa" and ruta_previa == "accion" and historial:
            info = {**info, "ruta": "accion", "motivo": "continúa una conversación de acción"}
        detalle = {}
        try:
            if info["ruta"] == "riesgo":
                texto, detalle = self._ruta_riesgo(mensaje, info)
            elif info["ruta"] == "informativa":
                texto, detalle = self._ruta_informativa(mensaje)
            else:
                texto, detalle = self._ruta_accion(mensaje, historial)
            error = None
        except Exception as e:  # noqa: BLE001 - el usuario siempre recibe una respuesta, y el error queda registrado
            texto = "Tuve un problema técnico al procesar su solicitud. Un agente humano continuará con usted."
            error = f"{type(e).__name__}: {e}"
        salida = {"respuesta": texto, "ruta": info["ruta"], "pedidos": info["pedidos"], "modelo": self.modelo,
                  "segundos": round(time.perf_counter() - t0, 1), "error": error, **detalle}
        salida["decision"] = _ultima_decision(salida.get("llamadas", []))
        salida["etiqueta"] = _etiqueta(salida.get("llamadas", []))
        if self.registrar:
            with open(REGISTRO_JSONL, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), "mensaje": mensaje, **salida},
                                   ensure_ascii=False, default=str) + "\n")
        return salida


def _ultima_decision(llamadas):
    for c in reversed(llamadas):
        if c["herramienta"] == "verificar_elegibilidad" and not c["resultado"].get("error"):
            return c["resultado"]["decision"]
    return None


def _ultima_seccion(llamadas):
    for c in reversed(llamadas):
        if c["herramienta"] == "verificar_elegibilidad" and not c["resultado"].get("error"):
            return c["resultado"].get("seccion_politica")
    return None


def _requiere_evidencia(llamadas):
    for c in reversed(llamadas):
        if c["herramienta"] == "verificar_elegibilidad" and not c["resultado"].get("error"):
            return bool(c["resultado"].get("requiere_evidencia"))
    return False


def _etiqueta(llamadas):
    for c in llamadas:
        if c["herramienta"] == "generar_etiqueta_devolucion" and not c["resultado"].get("error"):
            return c["resultado"]["numero_guia"]
    return None


def imprimir(r):
    print(f"[ruta: {r['ruta']} | {r['segundos']} s" + (f" | iteraciones {r['iteraciones']}" if "iteraciones" in r else "") + "]")
    for c in r.get("llamadas", []):
        res = c["resultado"]
        resumen = res.get("decision") or res.get("numero_guia") or ("ERROR: " + res["mensaje"] if res.get("error") else "ok")
        print(f"  -> {c['herramienta']}({json.dumps(c['argumentos'], ensure_ascii=False)}) = {resumen}")
    if r.get("fragmentos"):
        print(f"  -> RAG: {', '.join(r['fragmentos'])}")
    print(r["respuesta"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mensaje", nargs="?")
    ap.add_argument("--modelo", default=MODELO_POR_DEFECTO)
    ap.add_argument("--interactivo", action="store_true")
    a = ap.parse_args()
    if not a.mensaje and not a.interactivo:
        ap.error("indica un mensaje o usa --interactivo")
    agente = Agente(a.modelo)
    if a.interactivo:
        historial, ruta_previa = [], None
        while True:
            try:
                m = input("\nCliente> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not m:
                continue
            r = agente.responder(m, historial, ruta_previa)
            imprimir(r)
            historial += [HumanMessage(m), AIMessage(r["respuesta"])]
            ruta_previa = r["ruta"]
    else:
        imprimir(agente.responder(a.mensaje))
