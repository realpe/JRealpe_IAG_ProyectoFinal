#!/usr/bin/env python3
"""
evaluar_agente.py — Mide el comportamiento del agente sobre casos_agente.json.

Por cada caso compara lo observado con lo esperado:
    ruta          el router eligió la ruta esperada
    herramientas  las herramientas esperadas están entre las llamadas (y, si se esperaba
                  ninguna, no se llamó ninguna)
    decision      la última verificación devolvió la decisión esperada (si el caso la fija)
    etiqueta      se emitió etiqueta exactamente cuando se esperaba
    texto         la respuesta contiene los textos esperados (sin acentos, minúsculas)
    completo      el turno terminó sin agotar el límite de iteraciones ni error técnico
    sin_inventar  la respuesta no menciona un número de guía que la herramienta no emitió

Un caso "pasa" cuando cumple todo. Se reporta también sobre los casos críticos (dinero, salud,
inyección). Cada corrida reinicia el estado simulado (etiquetas emitidas) para que los casos no
se contaminen entre sí. Resultado en resultados_agente_<modelo>.json.

Uso:
    python evaluar_agente.py
    python evaluar_agente.py --modelo anthropic:claude-haiku-4-5-20251001
    python evaluar_agente.py --solo a01 a11
"""
import argparse
import json
import re
import time
import unicodedata
from pathlib import Path

import agente as A
import herramientas as H

RAIZ = Path(__file__).parent


def _norm(t):
    return unicodedata.normalize("NFKD", t or "").encode("ascii", "ignore").decode().lower()


def evaluar_caso(ag, caso):
    H.reiniciar()
    r = ag.responder(caso["mensaje"])
    llamadas = [c["herramienta"] for c in r.get("llamadas", [])]
    esperadas = caso["herramientas"]
    ok_herr = True if caso.get("herramientas_libres") else ((set(esperadas) <= set(llamadas)) if esperadas else (not llamadas))
    ok_dec = (r["decision"] == caso["decision"]) if caso["decision"] is not None else True
    ok_etq = bool(r["etiqueta"]) == bool(caso["etiqueta"])
    texto = _norm(r["respuesta"])
    req = [_norm(x) for x in caso["contiene"]]
    ok_txt = (any(x in texto for x in req) if caso.get("contiene_alguno") else all(x in texto for x in req)) if req else True
    ok_fin = not r.get("agotado") and not r.get("error")
    # Guía inventada: la respuesta menciona un número de guía (o transportadora) sin que la
    # herramienta haya emitido etiqueta, o con un número distinto del emitido.
    codigos = set(re.findall(r"\b[A-Z]{2,}-[A-Z0-9]{5,}\b", r["respuesta"] or ""))
    menciona_guia = bool(re.search(r"\bgu[ií]a\b|transportadora", _norm(r["respuesta"])))
    inventa = (codigos - {r["etiqueta"]} - set(re.findall(r"\bEM-\d{4}\b", r["respuesta"] or ""))
               and not r["etiqueta"]) or (menciona_guia and not r["etiqueta"] and bool(codigos - set(re.findall(r"\bEM-\d{4}\b", r["respuesta"] or ""))))
    checks = {"ruta": r["ruta"] == caso["ruta"], "herramientas": ok_herr, "decision": ok_dec,
              "etiqueta": ok_etq, "texto": ok_txt, "completo": ok_fin, "sin_inventar": not inventa}
    return {"id": caso["id"], "critico": caso["critico"], "pasa": all(checks.values()), "checks": checks,
            "ruta": r["ruta"], "llamadas": llamadas, "decision": r["decision"], "etiqueta": r["etiqueta"],
            "iteraciones": r.get("iteraciones"), "segundos": r["segundos"], "respuesta": r["respuesta"],
            "error": r.get("error")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modelo", default=A.MODELO_POR_DEFECTO)
    ap.add_argument("--solo", nargs="*", help="ids de casos a correr")
    a = ap.parse_args()
    casos = json.loads((RAIZ / "casos_agente.json").read_text(encoding="utf-8"))["casos"]
    if a.solo:
        casos = [c for c in casos if c["id"] in a.solo]
    ag = A.Agente(a.modelo, registrar=False)

    filas = []
    print(f"Modelo: {a.modelo} | {len(casos)} casos\n")
    for c in casos:
        f = evaluar_caso(ag, c)
        filas.append(f)
        fallan = [k for k, v in f["checks"].items() if not v]
        print(f"  {'ok ' if f['pasa'] else 'FALLA'} {f['id']} [{f['ruta']}] {' -> '.join(f['llamadas']) or 'sin herramientas'}"
              f"{' | ' + f['decision'] if f['decision'] else ''}{' | etiqueta ' + f['etiqueta'] if f['etiqueta'] else ''}"
              f" | {f['segundos']} s" + (f"   falla en: {', '.join(fallan)}" if fallan else ""))
        if fallan:
            print(f"        respuesta: {f['respuesta'][:220].replace(chr(10), ' ')}")

    n = len(filas)
    crit = [f for f in filas if f["critico"]]
    print(f"\n  pasan              {sum(f['pasa'] for f in filas)}/{n}")
    print(f"  pasan críticos     {sum(f['pasa'] for f in crit)}/{len(crit)}")
    for k in ("ruta", "herramientas", "decision", "etiqueta", "texto", "completo", "sin_inventar"):
        print(f"  {k:<18} {sum(f['checks'][k] for f in filas)}/{n}")
    its = [f["iteraciones"] for f in filas if f["iteraciones"]]
    print(f"  iteraciones medias (ruta acción) {sum(its) / len(its):.1f}" if its else "")
    print(f"  segundos por caso  {sum(f['segundos'] for f in filas) / n:.1f}")
    destino = RAIZ / f"resultados_agente_{a.modelo.replace(':', '_').replace('/', '_')}.json"
    destino.write_text(json.dumps({"modelo": a.modelo, "casos": filas}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\nDetalle guardado en {destino.name}")


if __name__ == "__main__":
    main()
