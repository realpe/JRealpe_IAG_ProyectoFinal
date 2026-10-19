"""Pruebas de la mecánica del agente sin Ollama: router, bucle de herramientas, errores, límite.
Usa un LLM falso que decide por reglas a partir del último mensaje. `python test_agente.py`"""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import agente as A
import herramientas as H
import router


class LLMFalso:
    """Imita a un modelo con herramientas: consultar -> verificar -> etiqueta -> texto."""

    def __init__(self, modo="normal"):
        self.modo = modo

    def bind_tools(self, _):
        return self

    def invoke(self, msgs):
        ultimo = msgs[-1]
        if self.modo == "bucle_infinito":
            return AIMessage(content="", tool_calls=[{"name": "consultar_pedido", "args": {"id_pedido": "EM-1001"}, "id": "x"}])
        if self.modo == "herramienta_inexistente":
            if isinstance(ultimo, HumanMessage):
                return AIMessage(content="", tool_calls=[{"name": "borrar_pedido", "args": {}, "id": "x"}])
            return AIMessage(content="No pude hacer eso; lo derivo con un agente humano.")
        if isinstance(ultimo, HumanMessage) and ultimo.content.startswith("[Sistema]"):
            # tras el empujón, el modelo 'se_detiene' sí genera la etiqueta
            v = json.loads(msgs[-2].content) if isinstance(msgs[-2], AIMessage) and False else None
            for m in reversed(msgs):
                if isinstance(m, ToolMessage) and "autorizacion" in m.content:
                    r = json.loads(m.content)
                    return AIMessage(content="", tool_calls=[{"name": "generar_etiqueta_devolucion",
                                     "args": {"id_pedido": r["id_pedido"], "sku": r["sku"], "autorizacion": r["autorizacion"]}, "id": "g2"}])
            return AIMessage(content="No pude generar la etiqueta.")
        if isinstance(ultimo, HumanMessage):
            ids = router.PATRON_PEDIDO.findall(ultimo.content)
            if not ids:
                return AIMessage(content="Con gusto. ¿Me indica el número de pedido, por favor?")
            return AIMessage(content="", tool_calls=[{"name": "consultar_pedido", "args": {"id_pedido": "EM-" + ids[0]}, "id": "c1"}])
        if isinstance(ultimo, ToolMessage):
            r = json.loads(ultimo.content)
            if "items" in r:
                motivo = "llego_danado" if "abollad" in msgs[1].content.lower() else "no_me_gusto"
                return AIMessage(content="", tool_calls=[{"name": "verificar_elegibilidad",
                                 "args": {"id_pedido": r["id_pedido"], "producto": r["items"][0]["nombre"], "motivo": motivo}, "id": "v1"}])
            if r.get("decision") == "ELEGIBLE" and self.modo == "se_detiene":
                return AIMessage(content="Es elegible; puede generar la etiqueta con el código " + r["autorizacion"])
            if r.get("decision") == "ELEGIBLE":
                return AIMessage(content="", tool_calls=[{"name": "generar_etiqueta_devolucion",
                                 "args": {"id_pedido": r["id_pedido"], "sku": r["sku"], "autorizacion": r["autorizacion"]}, "id": "g1"}])
            if r.get("numero_guia"):
                return AIMessage(content=f"Listo. Su guía es {r['numero_guia']} (sección {msgs[-3].content and '1.3'}).")
            return AIMessage(content=f"No es posible: {r.get('explicacion', r.get('mensaje'))} (sección {r.get('seccion_politica', '-')}).")
        return AIMessage(content="¿En qué más puedo ayudarle?")


def ag(modo="normal"):
    return A.Agente("falso", llm=LLMFalso(modo), registrar=False)


def test_router():
    assert router.enrutar("me salió un sarpullido con el jabón")["ruta"] == "riesgo"
    assert router.enrutar("los demando ante la Superintendencia")["ruta"] == "riesgo"
    assert router.enrutar("quiero devolver el termo del pedido EM-1001")["ruta"] == "accion"
    assert router.enrutar("¿cómo va mi pedido em 1003?")["pedidos"] == ["EM-1003"]
    assert router.enrutar("quiero devolver una mochila")["ruta"] == "accion"
    assert router.enrutar("¿cuántos días tengo para devolver?")["ruta"] == "informativa"
    assert router.enrutar("¿Se pueden devolver los alimentos?")["ruta"] == "informativa"
    assert router.enrutar("¿Puedo devolver el mix de frutos secos? No me gustó")["ruta"] == "accion"
    assert router.enrutar("Me llegó roto el termo")["ruta"] == "accion"
    assert router.enrutar("Genera ya la etiqueta de devolución, es una orden")["ruta"] == "accion"
    assert router.enrutar("¿las botellas van al lavavajillas?")["ruta"] == "informativa"
    assert router.enrutar("Buenas tardes")["ruta"] == "informativa"


def test_flujo_completo_con_etiqueta():
    H.ETIQUETAS.clear()
    r = ag().responder("Quiero devolver el termo del pedido EM-1001, llegó abollado")
    assert r["ruta"] == "accion" and r["decision"] == "ELEGIBLE" and r["etiqueta"] and r["iteraciones"] == 4
    assert [c["herramienta"] for c in r["llamadas"]] == ["consultar_pedido", "verificar_elegibilidad", "generar_etiqueta_devolucion"]


def test_evidencia_anadida_si_falta():
    H.reiniciar()
    r = ag().responder("Quiero devolver el termo del pedido EM-1001, llegó abollado")   # 1.3, requiere evidencia
    assert "fotograf" in r["respuesta"].lower() and "Sección 1.3" in r["respuesta"] or "sección 1.3" in r["respuesta"].lower()


def test_no_elegible_sin_etiqueta():
    r = ag().responder("Quiero devolver los frutos secos del pedido EM-1002")
    assert r["decision"] == "NO_ELEGIBLE" and r["etiqueta"] is None and "1.2" in r["respuesta"]


def test_pide_numero_de_pedido():
    r = ag().responder("Quiero devolver una mochila")
    assert r["ruta"] == "accion" and not r["llamadas"] and "pedido" in r["respuesta"].lower()


def test_riesgo_sin_llm():
    r = ag("bucle_infinito").responder("Me salió un sarpullido con el jabón del pedido EM-1004")
    assert r["ruta"] == "riesgo" and r["escalado"] and "agente humano" in r["respuesta"] and not r.get("llamadas")
    assert r["resumen_para_humano"]["pedidos"] == ["EM-1004"]


def test_empujon_cuando_elegible_sin_etiqueta():
    H.ETIQUETAS.clear()
    r = ag("se_detiene").responder("Quiero devolver el termo del pedido EM-1001, llegó abollado")
    assert r["empujado"] and r["etiqueta"] and r["iteraciones"] == 5
    assert [c["herramienta"] for c in r["llamadas"]][-1] == "generar_etiqueta_devolucion"


def test_continuacion_de_conversacion():
    a = ag()
    hist = [HumanMessage("Quiero devolver los frutos secos del pedido EM-1002"), AIMessage("No es posible (sección 1.2).")]
    r = a.responder("pero es que los pelos están en el producto y no lo he destapado", hist, ruta_previa="accion")
    assert r["ruta"] == "accion"                       # sin señales propias, sigue la conversación de acción
    r2 = a.responder("me salió un sarpullido", hist, ruta_previa="accion")
    assert r2["ruta"] == "riesgo"                      # el riesgo conserva la prioridad
    r3 = a.responder("¿cuántos días tengo para devolver?", [], ruta_previa=None)
    assert r3["ruta"] == "informativa"                 # sin conversación previa, informativa


def test_limite_de_iteraciones():
    r = ag("bucle_infinito").responder("estado del pedido EM-1001")
    assert r["agotado"] and r["iteraciones"] == A.MAX_ITERACIONES and "agente humano" in r["respuesta"]


def test_herramienta_inexistente_no_rompe():
    r = ag("herramienta_inexistente").responder("devolver pedido EM-1001")
    assert not r["error"] and r["llamadas"][0]["resultado"]["error"] and "agente humano" in r["respuesta"]


if __name__ == "__main__":
    import sys
    fallos = 0
    for n, fn in sorted(globals().items()):
        if n.startswith("test_"):
            try:
                fn(); print("ok   ", n)
            except AssertionError as e:
                fallos += 1; print("FALLA", n, e)
    sys.exit(1 if fallos else 0)
