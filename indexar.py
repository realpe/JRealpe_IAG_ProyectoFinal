#!/usr/bin/env python3
"""
indexar.py — Construye o actualiza el índice vectorial (ChromaDB) de la base de conocimiento.

Qué hace, en orden:
  1. Construye los fragmentos con base_conocimiento.py (segmentación por tipo + fusión F).
  2. Lee del índice los ids y hashes que ya existen.
  3. Compara: fragmentos nuevos o con hash distinto -> se embeben y se hacen upsert;
     ids que ya no existen en el corpus -> se borran; el resto no se toca.
  4. Imprime el resumen (cuántos añadió, actualizó, borró y dejó intactos).

Por qué incremental: embeber 54 fragmentos con bge-m3 toma segundos, pero con un corpus real
sería caro y lento reembeber todo por cambiar una política. El hash (SHA-256 del texto, versión
y enlace) decide qué cambió. Alternativa más simple: borrar y reconstruir todo (--reconstruir).

Uso:
    python indexar.py                 # indexación incremental
    python indexar.py --reconstruir   # borra la colección y reindexa todo
    python indexar.py --simulado      # embeddings falsos (sin Ollama), solo para probar la tubería
    python indexar.py --dry-run       # muestra qué haría sin escribir nada
"""
import argparse
import hashlib
import shutil
from pathlib import Path

import chromadb
from langchain_core.embeddings import Embeddings

import base_conocimiento as bc

RUTA_DB = Path(__file__).parent / "chroma_db"
COLECCION = "ecomarket"
MODELO_EMBEDDINGS = "bge-m3"
OLLAMA = "http://localhost:11434"
LOTE = 32  # fragmentos por llamada de embeddings


class EmbeddingsSimulados(Embeddings):
    """Vectores pseudoaleatorios deterministas derivados del texto. Sirven para probar la
    tubería (ids, hashes, upsert, borrado) sin Ollama. NO tienen significado semántico."""

    def __init__(self, dim=64):
        self.dim = dim

    def _vec(self, texto):
        semilla = hashlib.sha256(texto.encode("utf-8")).digest()
        out = []
        while len(out) < self.dim:
            semilla = hashlib.sha256(semilla).digest()
            out.extend((b - 127.5) / 127.5 for b in semilla)
        return out[: self.dim]

    def embed_documents(self, textos):
        return [self._vec(t) for t in textos]

    def embed_query(self, texto):
        return self._vec(texto)


def crear_embeddings(simulado):
    if simulado:
        return EmbeddingsSimulados()
    from langchain_ollama import OllamaEmbeddings
    # bge-m3 no usa prefijos de consulta/documento (medido en la Fase 1), por lo que
    # embed_documents y embed_query reciben el texto tal cual.
    return OllamaEmbeddings(model=MODELO_EMBEDDINGS, base_url=OLLAMA)


def abrir_coleccion(ruta_db, reconstruir):
    cliente = chromadb.PersistentClient(path=str(ruta_db))
    if reconstruir:
        try:
            cliente.delete_collection(COLECCION)
        except Exception:  # noqa: BLE001 - la colección puede no existir todavía
            pass
    # Distancia coseno: es la métrica con la que se midió el recall (vectores normalizados).
    # Se declara al crear la colección y no se puede cambiar después.
    return cliente.get_or_create_collection(COLECCION, metadata={"hnsw:space": "cosine"})


def plan(fragmentos, existentes):
    """existentes: {id: hash}. Devuelve (nuevos, cambiados, borrar, intactos)."""
    nuevos = [f for f in fragmentos if f["id"] not in existentes]
    cambiados = [f for f in fragmentos if f["id"] in existentes and existentes[f["id"]] != f["hash"]]
    ids_corpus = {f["id"] for f in fragmentos}
    borrar = sorted(i for i in existentes if i not in ids_corpus)
    intactos = len(fragmentos) - len(nuevos) - len(cambiados)
    return nuevos, cambiados, borrar, intactos


def indexar(ruta_db=RUTA_DB, simulado=False, reconstruir=False, dry_run=False):
    fragmentos = bc.construir_fragmentos()
    ids = [f["id"] for f in fragmentos]
    assert len(ids) == len(set(ids)), "ids duplicados: el upsert sobrescribiría fragmentos"

    col = abrir_coleccion(ruta_db, reconstruir and not dry_run)
    previo = col.get(include=["metadatas"])
    existentes = {i: m.get("hash") for i, m in zip(previo["ids"], previo["metadatas"])}
    nuevos, cambiados, borrar, intactos = plan(fragmentos, existentes)

    print(f"Corpus: {len(fragmentos)} fragmentos | índice actual: {len(existentes)}")
    print(f"  nuevos {len(nuevos)} | actualizados {len(cambiados)} | "
          f"a borrar {len(borrar)} | intactos {intactos}")
    for etiqueta, lista in (("nuevo", nuevos), ("actualizado", cambiados)):
        for f in lista[:5]:
            print(f"    {etiqueta}: {f['id']}")
    for i in borrar[:5]:
        print(f"    borrar: {i}")
    if dry_run:
        print("--dry-run: no se escribió nada.")
        return

    a_escribir = nuevos + cambiados
    if a_escribir:
        emb = crear_embeddings(simulado)
        for i in range(0, len(a_escribir), LOTE):
            lote = a_escribir[i:i + LOTE]
            vectores = emb.embed_documents([f["texto"] for f in lote])
            col.upsert(ids=[f["id"] for f in lote], embeddings=vectores,
                       documents=[f["texto"] for f in lote],
                       metadatas=[bc.metadatos(f) for f in lote])
    if borrar:
        col.delete(ids=borrar)
    print(f"Listo. Índice con {col.count()} fragmentos en {ruta_db}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reconstruir", action="store_true", help="borra la colección y reindexa todo")
    ap.add_argument("--simulado", action="store_true", help="embeddings falsos, sin Ollama")
    ap.add_argument("--dry-run", action="store_true", help="solo muestra el plan")
    ap.add_argument("--ruta-db", default=str(RUTA_DB), help="carpeta de la base Chroma")
    a = ap.parse_args()
    indexar(Path(a.ruta_db), a.simulado, a.reconstruir, a.dry_run)
