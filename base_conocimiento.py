#!/usr/bin/env python3
"""
base_conocimiento.py — Construye los fragmentos finales de la base de conocimiento de EcoMarket.

Es la configuración elegida en la Fase 2: segmentación por tipo de documento + fusión de las
secciones 1.1 a 1.3 de políticas (F). El enlace FAQ -> política (E) no cambia los fragmentos;
se guarda como metadato (`politica_ref`) y se aplica al consultar (ver rag_ecomarket.py).

Este módulo REUTILIZA las funciones de medir_segmentacion.py y medir_embeddings.py, las mismas
con las que se midió el recall. Así los fragmentos que se indexan son exactamente los que se
evaluaron, sin una segunda implementación que pueda desviarse. La alternativa (copiar el código
aquí) deja el módulo autocontenido, pero obliga a mantener dos copias sincronizadas.

La construcción es determinista y no usa ningún modelo de lenguaje.

Cada fragmento es un diccionario:
    id           identificador estable, p. ej. 'politicas::1.1-1.3', 'faq::faq-02', 'catalogo::ECO-BOT-500'
    texto        texto que se embebe y que verá el modelo generador
    fuente       politicas | catalogo | faq | guia
    version      versión del documento de origen
    claves       secciones de políticas que cubre el fragmento, separadas por coma ('1.1,1.2,1.3')
    politica_ref (solo FAQ) secciones de políticas que gobiernan la respuesta ('1.2,1.3')
    hash         SHA-256 de lo que determina el contenido indexado; detecta qué cambió
"""
import csv
import hashlib
import json
import re
import unicodedata

import medir_segmentacion as ms

RAIZ = ms.RAIZ
CORPUS = RAIZ / "corpus"


def _version_politicas():
    m = re.search(r"Versión\s+([\d.]+)", (CORPUS / "politicas.md").read_text(encoding="utf-8"))
    return m.group(1) if m else "sin-version"


def _version_guia():
    m = re.search(r"Versión\s+([\d.]+)", (CORPUS / "guia_cuidado.md").read_text(encoding="utf-8"))
    return m.group(1) if m else "sin-version"


def _version_faq():
    return json.loads((CORPUS / "faq.json").read_text(encoding="utf-8")).get("version", "sin-version")


def _slug(texto):
    """'Acero inoxidable: botellas y termos' -> 'acero-inoxidable-botellas-y-termos'."""
    s = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _skus():
    with open(CORPUS / "catalogo_productos.csv", encoding="utf-8", newline="") as f:
        return [fila["sku"] for fila in csv.DictReader(f)]


def _hash(texto, version, politica_ref):
    # Se incluyen la versión y el enlace: si solo cambia la tabla de enlaces, el fragmento
    # debe reindexarse aunque su texto sea idéntico (el metadato es parte del contenido).
    base = f"{version}\x1f{politica_ref}\x1f{texto}"
    return hashlib.sha256(base.encode("utf-8")).hexdigest()


def construir_fragmentos():
    """Fragmentos finales (F aplicada), con id, metadatos y hash. Orden estable."""
    frags, _ = ms.construir(fusion=True, preguntas=False)
    versiones = {"politicas": _version_politicas(), "faq": _version_faq(),
                 "guia": _version_guia(), "catalogo": "sin-version"}
    skus = iter(_skus())
    contador_guia = {}
    salida = []
    for f in frags:
        fuente = f["fuente"]
        if fuente == "politicas":
            claves = f["claves"]
            id_ = "politicas::" + (claves[0] if len(claves) == 1 else f"{claves[0]}-{claves[-1]}")
            ref = ""
        elif fuente == "catalogo":
            id_, claves, ref = f"catalogo::{next(skus)}", [], ""
        elif fuente == "faq":
            id_, claves = f"faq::{f['id']}", []
            ref = ",".join(ms.REFS_FAQ.get(f["id"], []))
        else:  # guia: sección + número de párrafo dentro de la sección
            seccion = _slug(f["texto"].split("\n", 1)[0].split(" > ", 1)[1])
            n = contador_guia[seccion] = contador_guia.get(seccion, 0) + 1
            id_, claves, ref = f"guia::{seccion}::{n}", [], ""
        salida.append({
            "id": id_, "texto": f["texto"], "fuente": fuente, "version": versiones[fuente],
            "claves": ",".join(c for c in claves if c), "politica_ref": ref,
            "hash": _hash(f["texto"], versiones[fuente], ref),
        })
    return salida


def metadatos(frag):
    """Metadatos que se guardan en Chroma (solo str/int/float/bool; no admite listas ni None)."""
    return {k: frag[k] for k in ("fuente", "version", "claves", "politica_ref", "hash")}


if __name__ == "__main__":
    fr = construir_fragmentos()
    por_fuente = {}
    for f in fr:
        por_fuente[f["fuente"]] = por_fuente.get(f["fuente"], 0) + 1
    print(f"{len(fr)} fragmentos: {por_fuente}")
    for f in fr[:8]:
        print(f"  {f['id']:<40} v{f['version']:<5} ref={f['politica_ref'] or '-':<8} {f['hash'][:10]}")
