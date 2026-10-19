#!/usr/bin/env python3
"""
medir_segmentacion.py — Evidencia de la Fase 2 (Taller 2): estrategias de segmentación y
recuperación sobre el corpus de EcoMarket, con el modelo de embedding ya elegido (bge-m3).

Motivación (resultados de medir_embeddings.py): oro-01 y oro-04 fallan con los tres modelos.
El recuperador prefiere los fragmentos de la FAQ (redactados como pregunta) sobre los de
políticas (redactados como norma), y la excepción por producto dañado (1.3) queda lejos de
la restricción por categoría (1.2). Este script mide tres intervenciones y sus combinaciones:

  F  fusión      : 1.1 + 1.2 + 1.3 se indexan como UN fragmento de devoluciones
  E  enlace      : cada par de la FAQ declara la sección de políticas que lo gobierna; si el
                   recuperador trae ese par, la sección se añade al contexto
  P  preguntas   : los fragmentos de políticas se INDEXAN con frases típicas de cliente
                   (doc2query). El texto evaluado y enviado al LLM es siempre el original.

Se evalúa sobre conjunto_oro.json (con el que se diseñaron) y sobre conjunto_validacion.json
(escrito antes de diseñar las estrategias). Solo una mejora que aparece en los dos cuenta como
evidencia de que generaliza.

Uso:
    python medir_segmentacion.py                  # bge-m3
    python medir_segmentacion.py --modelo nomic-embed-text
    python medir_segmentacion.py --simulado       # sin Ollama: solo valida la tubería
"""
import argparse
import copy
import json
import re

import numpy as np
import requests

import medir_embeddings as me

RAIZ = me.RAIZ
K_MAX = me.K_MAX

# Sección de políticas que gobierna cada pregunta de la FAQ (estrategia E). La tabla vive en
# corpus/enlaces_faq_politica.json: la FAQ la redacta mercadeo y el enlace lo mantiene el área
# dueña de las políticas, de modo que faq.json conserva su forma original.
REFS_FAQ = json.loads((RAIZ / "corpus" / "enlaces_faq_politica.json")
                      .read_text(encoding="utf-8"))["enlaces"]

# Frases de cliente por sección (estrategia P). Solo alimentan el embedding del fragmento.
PREGUNTAS_TIPICAS = {
    "1.1": ["¿Cuántos días tengo para devolver un producto?",
            "¿Hasta cuándo puedo pedir una devolución?"],
    "1.2": ["¿Qué productos se pueden devolver y cuáles no?",
            "¿Aceptan devolución de alimentos o de artículos de aseo ya abiertos?"],
    "1.3": ["¿Qué hago si el producto llegó roto, dañado, incompleto o defectuoso?",
            "¿Puedo devolver cualquier producto que venía en mal estado aunque ya lo haya abierto?"],
    "1.4": ["¿Cuándo interviene una persona en lugar del asistente?",
            "Tuve un problema de salud con el producto, voy a poner una queja formal o el reclamo es por un valor alto."],
    "2": ["¿Cuánto demora el envío y desde qué valor es gratis?",
          "¿Me garantizan una fecha exacta de entrega?"],
    "2.1": ["Mi pedido va con retraso, ¿qué opciones tengo?",
            "¿Me dan un cupón o una compensación por la demora?"],
    "3": ["¿Cómo debe hablarle el asistente al cliente?"],
    "4": ["¿Cuándo se pasa el caso a un agente humano?"],
}

PREFIJOS_OVERRIDE = {}   # modelo -> (prefijo_consulta, prefijo_documento); lo llena --instruccion-generica


def prefijos(modelo):
    return PREFIJOS_OVERRIDE.get(modelo) or me.prefijos_de(modelo)


ESTRATEGIAS = [  # nombre, fusión, enlace, preguntas
    ("base", False, False, False),
    ("F fusión", True, False, False),
    ("E enlace", False, True, False),
    ("P preguntas", False, False, True),
    ("F+E", True, True, False),
    ("F+P", True, False, True),
    ("F+E+P", True, True, True),
]


# ------------------------------------------------------------------ construcción del índice
def clave(texto):
    """Número de sección de un fragmento de políticas ('1.3', '2', ...), leído de su encabezado."""
    hallazgos = re.findall(r">\s*(\d+(?:\.\d+)?)[. ]", texto.split("\n", 1)[0])
    return hallazgos[-1] if hallazgos else None


def fragmentos_base():
    c = RAIZ / "corpus"
    pol = me.fragmentar_politicas(c / "politicas.md")
    for f in pol:
        f["claves"] = [clave(f["texto"])]
    faq = me.fragmentar_faq(c / "faq.json")
    ids = [p["id"] for p in json.loads((c / "faq.json").read_text(encoding="utf-8"))["preguntas"]]
    for f, i in zip(faq, ids):
        f["id"] = i
    return pol + me.fragmentar_catalogo(c / "catalogo_productos.csv") + faq \
        + me.fragmentar_guia(c / "guia_cuidado.md")


def fusionar(frags, claves=("1.1", "1.2", "1.3")):
    """Reemplaza los fragmentos de las secciones indicadas por uno solo, en la posición del
    primero. Conserva el subtítulo de cada sección dentro del texto fusionado."""
    partes = [f for f in frags if f["fuente"] == "politicas" and f["claves"][0] in claves]
    cuerpos = []
    for f in partes:
        titulo, cuerpo = f["texto"].split("\n", 1)
        cuerpos.append(f"{titulo.split(' > ')[-1]}\n{cuerpo}")
    unido = {"fuente": "politicas", "claves": list(claves),
             "texto": "Políticas EcoMarket > 1. Política de devoluciones > 1.1 a 1.3 "
                      "(plazo, elegibilidad por categoría y excepción por producto dañado)\n"
                      + "\n\n".join(cuerpos)}
    salida, puesto = [], False
    for f in frags:
        if f in partes:
            if not puesto:
                salida.append(unido)
                puesto = True
        else:
            salida.append(f)
    return salida


def construir(fusion, preguntas):
    frags = copy.deepcopy(fragmentos_base())
    if fusion:
        frags = fusionar(frags)
    for f in frags:
        f["indexado"] = f["texto"]
        if preguntas and f["fuente"] == "politicas":
            frases = [q for c in f["claves"] for q in PREGUNTAS_TIPICAS.get(c, [])]
            titulo, cuerpo = f["texto"].split("\n", 1)
            f["indexado"] = f"{titulo}\nPreguntas típicas de clientes: {' '.join(frases)}\n{cuerpo}"
    mapa = {c: i for i, f in enumerate(frags) if f["fuente"] == "politicas" for c in f["claves"]}
    return frags, mapa


def seleccionar(orden, k, frags, mapa, enlace):
    """Fragmentos que llegarían al LLM con k cupos de similitud. Con enlace, cada FAQ
    recuperada arrastra la sección de políticas que la gobierna."""
    sel = list(orden[:k])
    if enlace:
        for i in list(sel):
            if frags[i]["fuente"] == "faq":
                for c in REFS_FAQ.get(frags[i]["id"], []):
                    j = mapa.get(c)
                    if j is not None and j not in sel:
                        sel.append(j)
    return sel


# ------------------------------------------------------------------ evaluación
def evaluar(frags, mapa, enlace, preguntas, modelo, embed, cache):
    q_pre, d_pre = prefijos(modelo)

    def vectores(textos, tipo):
        faltan = [t for t in dict.fromkeys(textos) if (modelo, tipo, t) not in cache]
        if faltan:
            for t, v in zip(faltan, embed(modelo, faltan, tipo)):
                cache[(modelo, tipo, t)] = v
        return np.array([cache[(modelo, tipo, t)] for t in textos])

    matriz = vectores([d_pre + f["indexado"] for f in frags], "document")
    norm = [me.sin_acentos(f["texto"]) for f in frags]   # se evalúa el texto ORIGINAL
    detalle = []
    for p in preguntas:
        v = vectores([q_pre + p["pregunta"]], "query")[0]
        orden = list(np.argsort(-(matriz @ v))[:K_MAX])
        k_min = None
        for k in range(1, K_MAX + 1):
            union = " ".join(norm[j] for j in seleccionar(orden, k, frags, mapa, enlace))
            if all(me.contiene(union, r) for r in p["requiere"]):
                k_min = k
                break
        sel3 = seleccionar(orden, 3, frags, mapa, enlace)
        detalle.append({"id": p["id"], "critico": p["critico"], "k_minimo": k_min,
                        "ctx3": len(sel3),
                        "chars3": sum(len(frags[j]["texto"]) for j in sel3)})

    def recall(k, criticas=False):
        base = [d for d in detalle if d["critico"] or not criticas]
        return sum(1 for d in base if d["k_minimo"] is not None and d["k_minimo"] <= k) / len(base)

    return {"recall": {f"@{k}": round(recall(k), 3) for k in me.KS},
            "recall_criticas@3": round(recall(3, True), 3),
            "ctx_medio@3": round(float(np.mean([d["ctx3"] for d in detalle])), 2),
            "chars_ctx@3": round(float(np.mean([d["chars3"] for d in detalle]))),
            "fallan@3": [d["id"] for d in detalle
                         if d["k_minimo"] is None or d["k_minimo"] > 3],
            "fragmentos_indice": len(frags), "detalle": detalle}


def imprimir(titulo, filas):
    print(f"\n{titulo}")
    print(f"{'estrategia':<14}{'R@1':>6}{'R@3':>6}{'R@5':>6}{'crít@3':>8}{'ctx@3':>7}  fallan@3")
    print("-" * 78)
    for nombre, r in filas:
        print(f"{nombre:<14}{r['recall']['@1']:>6.2f}{r['recall']['@3']:>6.2f}"
              f"{r['recall']['@5']:>6.2f}{r['recall_criticas@3']:>8.2f}{r['ctx_medio@3']:>7.2f}  "
              f"{' '.join(i[-2:] for i in r['fallan@3'])}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--modelo", default="bge-m3")
    ap.add_argument("--simulado", action="store_true")
    ap.add_argument("--instruccion-generica", action="store_true",
                    help="qwen3-embedding: usar la instrucción genérica del model card (control)")
    args = ap.parse_args()

    conjuntos = {
        "ORO (18 preguntas; con el que se diseñaron las estrategias)": "conjunto_oro.json",
        "VALIDACIÓN (8 preguntas; escrito antes de diseñarlas)": "conjunto_validacion.json",
    }
    preguntas = {t: json.loads((RAIZ / a).read_text(encoding="utf-8"))["preguntas"]
                 for t, a in conjuntos.items()}

    if args.simulado:
        modelo, embed = "simulado-hash", me.embed_simulado
    elif args.modelo.startswith("voyage:"):
        if not me.os.environ.get("VOYAGE_API_KEY"):
            raise SystemExit("Falta la variable de entorno VOYAGE_API_KEY")
        modelo, embed = args.modelo, me.embed_voyage
    else:
        try:
            instalados = me.modelos_instalados()
        except requests.RequestException as e:
            raise SystemExit(f"No se pudo consultar Ollama en {me.OLLAMA}: {e}")
        modelo = next((n for n in instalados if n == args.modelo or n.startswith(args.modelo + ":")), None)
        if modelo is None:
            raise SystemExit(f"{args.modelo} no está instalado (ollama pull {args.modelo})")
        embed = me.embed_ollama
    generica = args.instruccion_generica and "qwen3-embedding" in modelo
    if generica:
        PREFIJOS_OVERRIDE[modelo] = (me.INSTRUCCION_GENERICA, "")
    print(f"Modelo: {modelo}" + (" [instr. genérica]" if generica else ""))

    cache, salida = {}, {}
    construidas = [(n, construir(f, p), e) for n, f, e, p in ESTRATEGIAS]

    # Precálculo: todos los textos de todas las estrategias y todas las preguntas se embeben
    # en una sola pasada por tipo. Con una API de tope bajo de solicitudes (cuenta gratuita de
    # Voyage) esto reduce decenas de llamadas a dos; con Ollama el resultado es el mismo.
    q_pre, d_pre = prefijos(modelo)
    docs = list(dict.fromkeys(d_pre + f["indexado"] for _, (fr, _m), _e in construidas for f in fr))
    qs = list(dict.fromkeys(q_pre + p["pregunta"] for lista in preguntas.values() for p in lista))
    for tipo, textos in (("document", docs), ("query", qs)):
        for t, v in zip(textos, embed(modelo, textos, tipo)):
            cache[(modelo, tipo, t)] = v

    for nombre, (frags, mapa), enlace in construidas:
        for titulo, lista in preguntas.items():
            salida.setdefault(titulo, []).append(
                (nombre, evaluar(frags, mapa, enlace, lista, modelo, embed, cache)))
    for titulo, filas in salida.items():
        imprimir(titulo, filas)
    if args.simulado:
        print("\n(modo simulado: no se escribe ningún archivo de resultados)")
        return

    # Con un modelo de API el JSON no debe pisar el resultado de bge-m3 ya obtenido
    destino = "resultados_segmentacion.json" if ":" not in modelo or modelo.endswith(":latest") \
        else f"resultados_segmentacion_{modelo.replace(':', '_')}.json"
    if generica:
        destino = destino.replace(".json", "_generica.json")
    (RAIZ / destino).write_text(
        json.dumps({"modelo": modelo + (" [instr. genérica]" if generica else ""), "tokens_api": me.TOKENS_API.get(modelo, 0),
                    "resultados": {t: dict(f) for t, f in salida.items()}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetalle guardado en {destino}")


if __name__ == "__main__":
    main()
