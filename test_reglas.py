"""Pruebas de reglas_devolucion.py: un caso por regla, con la sección esperada. `python -m pytest -q` o `python test_reglas.py`."""
from datetime import date
from pathlib import Path

import reglas_devolucion as r

HOY = date(2026, 10, 20)
P = r.cargar_pedidos(Path(__file__).parent / "datos" / "pedidos.json", HOY)


def it(pid, i=0):
    return P[pid]["items"][i]


def test_duradero_en_plazo_elegible():
    d = r.evaluar(P["EM-1001"], it("EM-1001"), "no_me_gusto", hoy=HOY)
    assert (d["decision"], d["seccion_politica"]) == ("ELEGIBLE", "1.2")


def test_dano_reportado_a_tiempo_prima_sobre_categoria():
    d = r.evaluar(P["EM-1002"], it("EM-1002"), "llego_danado", hoy=HOY)   # perecedero, 2 días
    assert (d["decision"], d["seccion_politica"]) == ("ELEGIBLE", "1.3") and d["requiere_evidencia"]


def test_perecedero_sin_dano_no_elegible():
    d = r.evaluar(P["EM-1002"], it("EM-1002"), "no_me_gusto", hoy=HOY)
    assert (d["decision"], d["seccion_politica"]) == ("NO_ELEGIBLE", "1.2")


def test_dano_fuera_de_72h_cae_a_regla_general():
    d = r.evaluar(P["EM-1008"], it("EM-1008"), "llego_danado", hoy=HOY)   # perecedero, 6 días
    assert (d["decision"], d["seccion_politica"]) == ("NO_ELEGIBLE", "1.2") and "72" in d["explicacion"]


def test_fuera_de_plazo_escala():
    d = r.evaluar(P["EM-1003"], it("EM-1003"), "no_me_gusto", hoy=HOY)    # 45 días
    assert (d["decision"], d["seccion_politica"]) == ("ESCALAR", "1.4")


def test_higiene_depende_del_empaque():
    a = r.evaluar(P["EM-1004"], it("EM-1004"), "no_me_gusto", empaque_abierto=False, hoy=HOY)
    b = r.evaluar(P["EM-1004"], it("EM-1004"), "no_me_gusto", empaque_abierto=True, hoy=HOY)
    assert a["decision"] == "ELEGIBLE" and b["decision"] == "NO_ELEGIBLE"


def test_higiene_abierta_pero_danada_elegible():
    d = r.evaluar(P["EM-1004"], it("EM-1004"), "defectuoso", empaque_abierto=True, hoy=HOY)  # 10 días > 72 h
    assert d["decision"] == "NO_ELEGIBLE"
    d2 = r.evaluar({**P["EM-1004"], "fecha_entrega": HOY}, it("EM-1004"), "defectuoso", empaque_abierto=True, hoy=HOY)
    assert (d2["decision"], d2["seccion_politica"]) == ("ELEGIBLE", "1.3")


def test_reclamo_alto_escala_y_item_solo_no():
    total = sum(i["precio_cop"] for i in P["EM-1005"]["items"])
    d = r.evaluar(P["EM-1005"], it("EM-1005"), "no_me_gusto", hoy=HOY, valor_reclamo=total)
    assert (d["decision"], d["seccion_politica"]) == ("ESCALAR", "1.4")
    assert r.evaluar(P["EM-1005"], it("EM-1005"), "no_me_gusto", hoy=HOY)["decision"] == "ELEGIBLE"


def test_segunda_devolucion_escala():
    d = r.evaluar(P["EM-1006"], it("EM-1006"), "no_me_gusto", hoy=HOY)
    assert (d["decision"], d["seccion_politica"]) == ("ESCALAR", "1.4")


def test_salud_escala_siempre():
    d = r.evaluar(P["EM-1001"], it("EM-1001"), "reaccion_salud", hoy=HOY)
    assert (d["decision"], d["seccion_politica"]) == ("ESCALAR", "1.4")


def test_no_entregado():
    d = r.evaluar(P["EM-1007"], it("EM-1007"), "no_me_gusto", hoy=HOY)
    assert d["decision"] == "NO_ELEGIBLE" and d.get("accion_alterna") == "cancelar"


def test_motivo_invalido():
    try:
        r.evaluar(P["EM-1001"], it("EM-1001"), "capricho", hoy=HOY)
        assert False
    except ValueError:
        pass


if __name__ == "__main__":
    import sys
    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_"):
            try:
                fn(); print("ok   ", nombre)
            except AssertionError as e:
                fallos += 1; print("FALLA", nombre, e)
    sys.exit(1 if fallos else 0)
