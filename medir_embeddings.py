#!/usr/bin/env python3
"""
medir_embeddings.py — Evidencia de la Fase 1 (Taller 2): comparar modelos de embedding
sobre el corpus de EcoMarket con el conjunto oro.

Qué hace
--------
1. Segmenta los 4 documentos del corpus, cada uno con la estrategia que corresponde a su forma:
     politicas.md            -> un fragmento por sección/subsección (unidad de decisión)
     catalogo_productos.csv  -> un fragmento por fila (un SKU = una unidad)
     faq.json                -> un fragmento por par pregunta/respuesta
     guia_cuidado.md         -> un fragmento por párrafo, con su encabezado de sección
   Los fragmentos son IDÉNTICOS para todos los modelos: la única variable es el embedding.
2. Para cada modelo instalado en Ollama, indexa los fragmentos y recupera los top-k de cada
   pregunta del conjunto oro.
3. Una pregunta cuenta como acierto en k si TODOS los textos de `requiere` aparecen en la unión
   de los k primeros fragmentos (misma definición que fase4_rag/evaluar.py del Taller 1).
4. Imprime una tabla comparativa y guarda resultados_embeddings.json.

Uso
---
    ollama pull nomic-embed-text      # baseline del Taller 1 (~270 MB)
    ollama pull bge-m3                # multilingüe (~1.2 GB)
    python3 medir_embeddings.py
    python3 medir_embeddings.py --modelos nomic-embed-text bge-m3
    python3 medir_embeddings.py --simulado     # sin Ollama: valida solo la tubería

Modelos de API (propietarios): se piden con el prefijo del proveedor y la clave se toma de una
variable de entorno que se exporta en la terminal (nunca se escribe en archivos del repo):
    export VOYAGE_API_KEY="..."
    python3 medir_embeddings.py --modelos nomic-embed-text bge-m3 voyage:voyage-4 voyage:voyage-4-lite

Modelos locales más grandes (descarga uno a uno; ver tamaños con `ollama list`):
    ollama pull qwen3-embedding:0.6b   # 639 MB
    ollama pull qwen3-embedding:4b     # 2.5 GB
    ollama pull qwen3-embedding:8b     # 4.7 GB (el más grande del catálogo de embeddings)
    python3 medir_embeddings.py --modelos qwen3-embedding:0.6b qwen3-embedding:4b qwen3-embedding:8b

Los resultados se FUSIONAN en resultados_embeddings.json: medir modelos nuevos conserva los
anteriores (si se repite un modelo, su entrada se reemplaza).
Los modelos que no estén instalados (o sin clave) se omiten con un aviso (no se descargan solos).
"""
import argparse
import csv
import json
import os
import re
import time
import unicodedata
from pathlib import Path

import numpy as np
import requests

RAIZ = Path(__file__).parent
OLLAMA = "http://localhost:11434"
KS = (1, 3, 5)          # valores de k que se reportan
K_MAX = 10              # profundidad máxima explorada por pregunta
LOTE = 32               # fragmentos por llamada a /api/embed
LOTE_API = 128          # textos por solicitud a APIs externas (límite de Voyage: 1.000)
VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"
TOKENS_API = {}         # tokens facturables consumidos por modelo de API (para el eje de costo)
# USD por millón de tokens según docs.voyageai.com/docs/pricing (consultado 2026-09-30).
# Los primeros 200 M de tokens al mes son gratis en estos modelos.
PRECIO_USD_POR_M = {"voyage-4": 0.06, "voyage-4-lite": 0.02, "voyage-4-large": 0.12}

# Cada familia de modelos se entrena con convenciones distintas para consulta vs documento.
# Usar el prefijo equivocado degrada el recall sin dar ningún error, por eso queda declarado aquí
# y se guarda en el JSON de resultados.
#   nomic-embed-text : "search_query: " / "search_document: "  (model card de Nomic)
#   e5 (multilingual-e5-*) : "query: " / "passage: "
#   snowflake-arctic-embed2: "query: " solo en la consulta
#   bge-m3           : sin prefijos
#   qwen3-embedding  : la consulta lleva una instrucción de tarea; el documento va sin prefijo
#                      (formato del model card de Qwen3-Embedding)
#   embeddinggemma   : "task: search result | query: " / "title: none | text: "
#                      (formato del model card de EmbeddingGemma)
#   nomic-embed-text-v2-moe coincide con el patrón de nomic (mismos prefijos que la v1)
#   paraphrase-multilingual, granite-embedding: sin prefijos
# Instrucción de ejemplo del model card de Qwen3-Embedding, sin referencia a EcoMarket.
INSTRUCCION_GENERICA = ("Instruct: Given a web search query, retrieve relevant passages "
                        "that answer the query\nQuery: ")
PREFIJOS = [
    ("qwen3-embedding",
     "Instruct: Dada una pregunta de un cliente de EcoMarket, recuperar los pasajes de "
     "políticas, FAQ, catálogo o guía de cuidado que la responden\nQuery: ", ""),
    ("embeddinggemma", "task: search result | query: ", "title: none | text: "),
    ("nomic-embed-text", "search_query: ", "search_document: "),
    ("e5", "query: ", "passage: "),
    ("snowflake-arctic-embed2", "query: ", ""),
    ("bge-m3", "", ""),
]
MODELOS_POR_DEFECTO = ["nomic-embed-text", "bge-m3", "snowflake-arctic-embed2"]


def prefijos_de(modelo):
    for patron, q, d in PREFIJOS:
        if patron in modelo:
            return q, d
    return "", ""


# ------------------------------------------------------------------ segmentación
def fragmentar_politicas(ruta):
    """Un fragmento por sección o subsección. El encabezado completo (ruta de títulos) se
    antepone al texto: 'Tabla de categorías' sola no dice a qué política pertenece."""
    fragmentos, titulos, cuerpo = [], {}, []

    def cerrar():
        texto = "\n".join(cuerpo).strip()
        if texto:
            ruta_titulos = " > ".join(titulos[n] for n in sorted(titulos))
            fragmentos.append({"fuente": "politicas",
                               "texto": f"Políticas EcoMarket > {ruta_titulos}\n{texto}"})
        cuerpo.clear()

    dentro = False   # se ignora la cabecera meta anterior al primer '## '
    for linea in ruta.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(#{2,3})\s+(.*)", linea)
        if m:
            dentro = True
            cerrar()
            nivel = len(m.group(1))
            titulos = {n: t for n, t in titulos.items() if n < nivel}
            titulos[nivel] = m.group(2).strip()
        elif dentro and linea.strip() != "---":
            cuerpo.append(linea)
    cerrar()
    return fragmentos


def fragmentar_catalogo(ruta):
    """Una fila = un producto. Se serializa como 'campo: valor' para que el embedding vea
    los nombres de columna (p. ej. 'apto_lavavajillas: No') y no solo los valores sueltos."""
    fragmentos = []
    with open(ruta, encoding="utf-8", newline="") as f:
        for fila in csv.DictReader(f):
            partes = [f"{campo}: {valor}" for campo, valor in fila.items()]
            fragmentos.append({"fuente": "catalogo",
                               "texto": f"Producto del catálogo EcoMarket — " + "; ".join(partes)})
    return fragmentos


def fragmentar_faq(ruta):
    """Un par pregunta/respuesta por fragmento; la pregunta va incluida porque es el mejor
    ancla semántica para otra pregunta de cliente."""
    datos = json.loads(ruta.read_text(encoding="utf-8"))
    return [{"fuente": "faq",
             "texto": f"Pregunta frecuente ({p['tema']}): {p['pregunta']}\nRespuesta: {p['respuesta']}"}
            for p in datos["preguntas"]]


def fragmentar_guia(ruta):
    """Texto narrativo sin estructura de campos: un párrafo por fragmento, precedido del
    encabezado de su sección para que 'el termo' no quede huérfano de contexto."""
    fragmentos, seccion = [], ""
    for bloque in re.split(r"\n\s*\n", ruta.read_text(encoding="utf-8")):
        bloque = bloque.strip()
        if bloque.startswith("## "):
            seccion = bloque[3:].strip()
        elif seccion and bloque:
            fragmentos.append({"fuente": "guia",
                               "texto": f"Guía de cuidado > {seccion}\n{bloque}"})
    return fragmentos


def cargar_fragmentos():
    c = RAIZ / "corpus"
    return (fragmentar_politicas(c / "politicas.md")
            + fragmentar_catalogo(c / "catalogo_productos.csv")
            + fragmentar_faq(c / "faq.json")
            + fragmentar_guia(c / "guia_cuidado.md"))


# ------------------------------------------------------------------ embeddings
def normalizar(m):
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)   # con vectores unitarios, coseno = producto punto


def embed_ollama(modelo, textos, tipo="document"):
    """Devuelve matriz normalizada (n x dim). /api/embed acepta listas: un viaje por lote.
    `tipo` ('query'/'document') lo usan los modelos de API; en Ollama la diferencia se marca
    con prefijos de texto, que aplica el llamador."""
    salida = []
    for i in range(0, len(textos), LOTE):
        r = requests.post(f"{OLLAMA}/api/embed",
                          json={"model": modelo, "input": textos[i:i + LOTE]}, timeout=300)
        r.raise_for_status()
        salida.extend(r.json()["embeddings"])
    return normalizar(np.array(salida, dtype=np.float32))


def embed_voyage(modelo, textos, tipo="document"):
    """Modelos de Voyage AI (recomendado por Anthropic; Claude no tiene modelo de embeddings).
    El modelo se pide como 'voyage:voyage-4'. La clave se lee de la variable de entorno
    VOYAGE_API_KEY y nunca se escribe en el código ni en el repo.
    Voyage distingue consulta de documento con el campo input_type en lugar de prefijos."""
    clave = os.environ["VOYAGE_API_KEY"]
    nombre = modelo.split(":", 1)[1]
    salida = []
    # Lotes grandes (hasta 1.000 textos por solicitud según la API): la cuenta gratuita sin
    # método de pago tiene un tope bajo de solicitudes por minuto, así que conviene enviar
    # pocas solicitudes y gordas en lugar de muchas pequeñas.
    for i in range(0, len(textos), LOTE_API):
        for intento in range(7):
            r = requests.post(VOYAGE_URL, headers={"Authorization": f"Bearer {clave}"},
                              json={"input": textos[i:i + LOTE_API], "model": nombre,
                                    "input_type": tipo}, timeout=120)
            if r.status_code == 429:
                # Respeta Retry-After si viene; si no, espera 20, 40, 60... s (la ventana del
                # límite es de un minuto, así que esperas cortas no sirven).
                espera = int(r.headers.get("Retry-After", 20 * (intento + 1)))
                print(f"  Voyage pide pausa (límite de tasa). Esperando {espera} s "
                      f"(intento {intento + 1}/7)...", flush=True)
                time.sleep(espera)
                continue
            r.raise_for_status()
            break
        else:
            raise RuntimeError("Voyage: límite de tasa tras 7 reintentos. Agregar un método "
                               "de pago en la cuenta sube el límite y conserva los 200 M "
                               "de tokens gratis.")
        cuerpo = r.json()
        salida.extend(d["embedding"] for d in sorted(cuerpo["data"], key=lambda d: d["index"]))
        TOKENS_API[modelo] = TOKENS_API.get(modelo, 0) + cuerpo.get("usage", {}).get("total_tokens", 0)
    return normalizar(np.array(salida, dtype=np.float32))


def embed_simulado(_modelo, textos, tipo="document"):
    """Bolsa de palabras hasheada. Solo sirve para comprobar que la tubería corre sin Ollama;
    sus números NO son evidencia de nada."""
    m = np.zeros((len(textos), 512), dtype=np.float32)
    for i, t in enumerate(textos):
        for palabra in re.findall(r"\w+", sin_acentos(t)):
            m[i, hash(palabra) % 512] += 1
    return normalizar(m)


def modelos_instalados():
    r = requests.get(f"{OLLAMA}/api/tags", timeout=10)
    r.raise_for_status()
    return {m["name"]: m.get("size", 0) for m in r.json()["models"]}


# ------------------------------------------------------------------ evaluación
def sin_acentos(s):
    return "".join(c for c in unicodedata.normalize("NFKD", s.lower())
                   if not unicodedata.combining(c))


def contiene(texto_norm, requerido):
    """Coincidencia sin acentos ni mayúsculas. Los requisitos puramente numéricos ('5', '12')
    se buscan como palabra completa; de lo contrario '12' acertaría con '120.000'."""
    r = sin_acentos(requerido)
    if r.isdigit():
        return re.search(rf"(?<![\d.]){re.escape(r)}(?![\d.])", texto_norm) is not None
    return r in texto_norm


def k_minimo(orden, fragmentos_norm, requiere):
    """Menor k en 1..K_MAX tal que la unión de los k primeros fragmentos contiene todos los
    textos requeridos; None si ni con K_MAX se logra."""
    for k in range(1, K_MAX + 1):
        union = " ".join(fragmentos_norm[j] for j in orden[:k])
        if all(contiene(union, r) for r in requiere):
            return k
    return None


def evaluar_modelo(nombre, fragmentos, preguntas, embed, etiqueta=None, prefijos=None):
    """`etiqueta` permite guardar una variante (p. ej. otra instrucción) bajo un nombre
    distinto sin tocar el modelo que se invoca; `prefijos` reemplaza los de PREFIJOS."""
    q_pre, d_pre = prefijos if prefijos else prefijos_de(nombre)
    t0 = time.perf_counter()
    matriz = embed(nombre, [d_pre + f["texto"] for f in fragmentos], "document")
    t_indice = time.perf_counter() - t0

    norm = [sin_acentos(f["texto"]) for f in fragmentos]
    detalle, latencias = [], []
    por_lote = nombre.startswith("voyage:")   # API con tope de solicitudes: todas las consultas juntas
    if por_lote:
        t0 = time.perf_counter()
        vq = embed(nombre, [q_pre + p["pregunta"] for p in preguntas], "query")
        latencias = [(time.perf_counter() - t0) * 1000 / len(preguntas)] * len(preguntas)
    for n, p in enumerate(preguntas):
        if por_lote:
            v = vq[n]          # latencia por consulta = tiempo del lote / nº de consultas
        else:
            t0 = time.perf_counter()
            v = embed(nombre, [q_pre + p["pregunta"]], "query")[0]
            latencias.append((time.perf_counter() - t0) * 1000)
        puntajes = matriz @ v
        orden = list(np.argsort(-puntajes)[:K_MAX])
        detalle.append({
            "id": p["id"], "critico": p["critico"],
            "k_minimo": k_minimo(orden, norm, p["requiere"]),
            "top3": [{"fuente": fragmentos[j]["fuente"], "puntaje": round(float(puntajes[j]), 3),
                      "inicio": fragmentos[j]["texto"][:70].replace("\n", " ")} for j in orden[:3]],
        })

    def recall(k, solo_criticas=False):
        base = [d for d in detalle if d["critico"] or not solo_criticas]
        return sum(1 for d in base if d["k_minimo"] is not None and d["k_minimo"] <= k) / len(base)

    return {
        "modelo": etiqueta or nombre, "dim": int(matriz.shape[1]), "fragmentos": len(fragmentos),
        "prefijo_consulta": q_pre, "prefijo_documento": d_pre,
        "recall": {f"@{k}": round(recall(k), 3) for k in KS},
        "recall_criticas": {f"@{k}": round(recall(k, True), 3) for k in KS},
        "ms_por_consulta": round(float(np.mean(latencias)), 1),
        "s_indexado": round(t_indice, 1),
        "detalle": detalle,
    }


def imprimir(resultados):
    print(f"\n{'modelo':<28}{'dim':>5}{'R@1':>7}{'R@3':>7}{'R@5':>7}"
          f"{'crít@3':>8}{'ms/q':>8}{'MB':>7}")
    print("-" * 77)
    for r in resultados:
        print(f"{r['modelo']:<28}{r['dim']:>5}{r['recall']['@1']:>7.2f}{r['recall']['@3']:>7.2f}"
              f"{r['recall']['@5']:>7.2f}{r['recall_criticas']['@3']:>8.2f}"
              f"{r['ms_por_consulta']:>8.1f}{(r['mb'] if r.get('mb') is not None else 'API'):>7}")
    print("\nPreguntas que fallan en k=3 (✗) por modelo — las críticas van marcadas con *")
    ids = [d["id"] for d in resultados[0]["detalle"]]
    print(f"{'':<28}" + " ".join(i[-2:] for i in ids))
    for r in resultados:
        marcas = []
        for d in r["detalle"]:
            ok = d["k_minimo"] is not None and d["k_minimo"] <= 3
            marcas.append(("✓" if ok else "✗") + ("*" if d["critico"] else " "))
        print(f"{r['modelo']:<28}" + " ".join(marcas))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--modelos", nargs="+", default=MODELOS_POR_DEFECTO)
    ap.add_argument("--simulado", action="store_true", help="embedder falso, sin Ollama")
    ap.add_argument("--instruccion-generica", action="store_true",
                    help="modelos qwen3-embedding: usar la instrucción genérica del model card en\n"
                         "lugar de la que menciona EcoMarket (control del efecto de la instrucción)")
    args = ap.parse_args()

    fragmentos = cargar_fragmentos()
    preguntas = json.loads((RAIZ / "conjunto_oro.json").read_text(encoding="utf-8"))["preguntas"]
    por_fuente = {}
    for f in fragmentos:
        por_fuente[f["fuente"]] = por_fuente.get(f["fuente"], 0) + 1
    print(f"Corpus: {len(fragmentos)} fragmentos {por_fuente} | {len(preguntas)} preguntas oro")

    resultados = []
    if args.simulado:
        r = evaluar_modelo("simulado-hash", fragmentos, preguntas, embed_simulado)
        r["mb"] = 0
        resultados.append(r)
    else:
        instalados = None   # se consulta a Ollama solo si se pide algún modelo local
        for nombre in args.modelos:
            if nombre.startswith("voyage:"):
                if not os.environ.get("VOYAGE_API_KEY"):
                    print(f"[omitido] {nombre}: falta la variable de entorno VOYAGE_API_KEY")
                    continue
                print(f"Evaluando {nombre} (API) ...")
                r = evaluar_modelo(nombre, fragmentos, preguntas, embed_voyage)
                r["mb"] = None
                r["tokens_api"] = TOKENS_API.get(nombre, 0)
                precio = PRECIO_USD_POR_M.get(nombre.split(":", 1)[1])
                r["usd_estimado_sin_tramo_gratis"] = (
                    round(r["tokens_api"] * precio / 1e6, 6) if precio is not None else None)
                print(f"  tokens consumidos: {r['tokens_api']}")
                resultados.append(r)
                continue
            if instalados is None:
                try:
                    instalados = modelos_instalados()
                except requests.RequestException as e:
                    raise SystemExit(f"No se pudo consultar Ollama en {OLLAMA}: {e}")
            etiqueta = next((n for n in instalados
                             if n == nombre or n.startswith(nombre + ":")), None)
            if etiqueta is None:
                print(f"[omitido] {nombre} no está instalado (ollama pull {nombre})")
                continue
            variante, pref = None, None
            if args.instruccion_generica and "qwen3-embedding" in etiqueta:
                # Control: la instrucción del model card, sin mencionar el dominio de EcoMarket
                variante, pref = f"{etiqueta} [instr. genérica]", (INSTRUCCION_GENERICA, "")
            print(f"Evaluando {variante or etiqueta} ...")
            r = evaluar_modelo(etiqueta, fragmentos, preguntas, embed_ollama, variante, pref)
            r["mb"] = round(instalados[etiqueta] / 1e6)
            resultados.append(r)

    if not resultados:
        raise SystemExit("Ningún modelo evaluado.")
    imprimir(resultados)
    if args.simulado:
        print("\n(modo simulado: no se escribe ningún archivo de resultados)")
        return
    # Fusiona con los resultados previos: medir modelos nuevos no debe borrar los ya medidos.
    destino = RAIZ / "resultados_embeddings.json"
    previos = []
    if destino.exists():
        try:
            previos = json.loads(destino.read_text(encoding="utf-8"))
        except ValueError:
            previos = []
    nuevos = {r["modelo"] for r in resultados}
    fusion = [r for r in previos if r["modelo"] not in nuevos] + resultados
    destino.write_text(json.dumps(fusion, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetalle guardado en {destino.name} ({len(fusion)} modelos en total)")


if __name__ == "__main__":
    main()
