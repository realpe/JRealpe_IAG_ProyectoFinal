"""Pruebas de herramientas.py (sin LLM ni Ollama). `python test_herramientas.py`"""
import herramientas as h


def test_consultar_existente_y_no():
    r = h.consultar_pedido.invoke({"id_pedido": "em-1001"})
    assert r["encontrado"] and r["items"][0]["sku"] == "ECO-TER-1L"
    assert not h.consultar_pedido.invoke({"id_pedido": "EM-9999"})["encontrado"]


def test_resolver_por_nombre_y_error_guiado():
    ok = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1004", "producto": "cepillo", "motivo": "no_me_gusto"})
    assert not ok["error"] and ok["sku"] == "ECO-CEP-BAM" and ok["decision"] == "ELEGIBLE"
    err = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1004", "producto": "termo", "motivo": "no_me_gusto"})
    assert err["error"] and len(err["productos_del_pedido"]) == 2


def test_motivo_invalido_guiado():
    r = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1001", "producto": "termo", "motivo": "capricho"})
    assert r["error"] and "motivos_validos" in r


def test_etiqueta_exige_autorizacion():
    sin = h.generar_etiqueta_devolucion.invoke({"id_pedido": "EM-1001", "sku": "ECO-TER-1L", "autorizacion": "INVENTADA"})
    assert sin["error"]
    v = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1001", "producto": "termo", "motivo": "no_me_gusto"})
    con = h.generar_etiqueta_devolucion.invoke({"id_pedido": "EM-1001", "sku": "ECO-TER-1L", "autorizacion": v["autorizacion"]})
    assert not con["error"] and con["numero_guia"].startswith("ECO-")
    otra = h.generar_etiqueta_devolucion.invoke({"id_pedido": "EM-1001", "sku": "ECO-TER-1L", "autorizacion": v["autorizacion"]})
    assert otra["numero_guia"] == con["numero_guia"] and "aviso" in otra


def test_no_elegible_no_da_autorizacion():
    v = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1002", "producto": "frutos secos", "motivo": "no_me_gusto"})
    assert v["decision"] == "NO_ELEGIBLE" and "autorizacion" not in v
    e = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1003", "producto": "mochila", "motivo": "no_me_gusto"})
    assert e["decision"] == "ESCALAR" and e["requiere_humano"]


def test_ambiguedad_no_adivina():
    r = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1005", "producto": "botella", "motivo": "no_me_gusto"})
    assert r["error"] and len(r["candidatos"]) == 2
    r = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1005", "producto": "la botella de 500 ml", "motivo": "no_me_gusto"})
    assert not r["error"] and r["sku"] == "ECO-BOT-500"


def test_autorizacion_de_otro_sku_no_sirve():
    v = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1005", "producto": "botella 500", "motivo": "no_me_gusto"})
    assert v["sku"] == "ECO-BOT-500"
    r = h.generar_etiqueta_devolucion.invoke({"id_pedido": "EM-1005", "sku": "ECO-TER-1L", "autorizacion": v["autorizacion"]})
    assert r["error"]


def test_fraccionamiento_del_reclamo():
    h.reiniciar()
    skus = ["botella 750", "botella de 500", "termo", "mochila", "vela", "café"]
    r = [h.verificar_elegibilidad.invoke({"id_pedido": "EM-1005", "producto": s, "motivo": "no_me_gusto"}) for s in skus]
    assert [x["decision"] for x in r[:5]] == ["ELEGIBLE"] * 5 and r[4]["valor_reclamo_cop"] == 476000
    assert r[5]["decision"] == "ESCALAR" and r[5]["valor_reclamo_cop"] == 531000   # el acumulado supera el tope

def test_pedido_completo_escala():
    h.reiniciar()
    r = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1005", "producto": "termo", "motivo": "no_me_gusto", "devolver_pedido_completo": True})
    assert r["decision"] == "ESCALAR" and r["valor_reclamo_cop"] == 531000


def test_tope_de_etiquetas_por_pedido():
    h.reiniciar()
    guias = []
    for s in ["botella 750", "botella de 500", "termo"]:
        v = h.verificar_elegibilidad.invoke({"id_pedido": "EM-1005", "producto": s, "motivo": "no_me_gusto"})
        guias.append(h.generar_etiqueta_devolucion.invoke({"id_pedido": "EM-1005", "sku": v["sku"], "autorizacion": v["autorizacion"]}))
    assert not guias[0]["error"] and not guias[1]["error"] and guias[2]["error"] and guias[2]["decision"] == "ESCALAR"


def test_registro():
    assert len(h.REGISTRO) >= 8 and {e["herramienta"] for e in h.REGISTRO} >= {"consultar_pedido", "verificar_elegibilidad", "generar_etiqueta_devolucion"}


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
