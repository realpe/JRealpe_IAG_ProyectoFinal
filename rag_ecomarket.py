#!/usr/bin/env python3
"""
rag_ecomarket.py — Sistema RAG del asistente de atención al cliente de EcoMarket (LangChain).

Flujo de una consulta (cada paso es un bloque marcado con "# [n]" en el código):

    pregunta
      [1] RECUPERAR   embedding de la pregunta (bge-m3) -> top-k fragmentos más similares en Chroma
      [2] ENLAZAR     (E) cada FAQ recuperada arrastra la sección de políticas que la gobierna
      [3] ARMAR       los fragmentos se numeran y se insertan en el prompt como CONTEXTO
      [4] GENERAR     qwen3:14b responde usando solo ese contexto
      -> respuesta + fragmentos usados (para poder citar y auditar)

Los pasos [1] a [4] se encadenan con LCEL (el operador `|` de LangChain). El índice se crea
antes con `python indexar.py`; este script solo lo consulta.

Uso:
    python rag_ecomarket.py "¿Puedo devolver un termo que me llegó abollado?"
    python rag_ecomarket.py "..." --mostrar-contexto          # ve qué fragmentos recibió el modelo
    python rag_ecomarket.py "..." --sin-enlace --k 5          # experimentos: apaga E, cambia k
    python rag_ecomarket.py "..." --prompt estricto           # cambia la plantilla del prompt
    python rag_ecomarket.py --interactivo                     # conversación en consola
"""
import argparse
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnablePassthrough

RUTA_DB = Path(__file__).parent / "chroma_db"
COLECCION = "ecomarket"
MODELO_EMBEDDINGS = "bge-m3"
MODELO_LLM = "qwen3:14b"       # el mismo generador del Taller 1
OLLAMA = "http://localhost:11434"
K = 3                          # fragmentos por similitud; el enlace puede añadir más

# ------------------------------------------------------------------------- prompts
# Plantillas intercambiables (--prompt). Todas comparten las variables {contexto} y {pregunta}.
# Decisiones de diseño del prompt "base":
#  - "Usa SOLO el contexto": evita que el modelo complete con conocimiento propio y se invente
#    políticas. Es la regla que convierte la recuperación en fundamento de la respuesta.
#  - "Si hay conflicto, la política prevalece": en el corpus la FAQ (redactada por mercadeo)
#    omite excepciones que sí están en la política (ej. producto dañado). La regla se
#    justifica con ese hallazgo de la Fase 2.
#  - Citar el id entre corchetes: permite auditar de dónde salió cada afirmación.
#  - Si el contexto no alcanza, decirlo y derivar a un agente humano: es mejor que responder mal
#    sobre devoluciones o reembolsos, donde un error tiene costo económico para el cliente.
PROMPTS = {
    "base": (
        "Eres el asistente de atención al cliente de EcoMarket, una tienda de productos sostenibles.\n"
        "Responde en español, con un tono cordial y claro, y trata siempre al cliente de usted.\n\n"
        "Reglas:\n"
        "1. Usa SOLO la información del CONTEXTO. No inventes políticas, plazos, precios ni productos.\n"
        "2. Si dos fragmentos se contradicen o la FAQ omite una excepción, prevalece el fragmento de "
        "políticas.\n"
        "3. Cita entre corchetes el id de cada fragmento en que te apoyas, por ejemplo [politicas::1.4].\n"
        "4. Si el contexto no contiene la respuesta, dilo y ofrece derivar al cliente con un agente "
        "humano. No adivines.\n\n"
        "CONTEXTO:\n{contexto}\n\n"
        "PREGUNTA DEL CLIENTE: {pregunta}"
    ),
    # Variante: obliga a copiar el id tal cual. Con el prompt base el modelo citó [politicas::1.3],
    # que no existe como id (el fragmento fusionado es 'politicas::1.1-1.3').
    "citas_exactas": (
        "Eres el asistente de atención al cliente de EcoMarket, una tienda de productos sostenibles.\n"
        "Responde en español, con un tono cordial y claro, y trata siempre al cliente de usted.\n\n"
        "Reglas:\n"
        "1. Usa SOLO la información del CONTEXTO. No inventes políticas, plazos, precios ni productos.\n"
        "2. Si dos fragmentos se contradicen o la FAQ omite una excepción, prevalece el fragmento de "
        "políticas.\n"
        "3. Cita entre corchetes el id EXACTO que aparece al inicio de cada fragmento, por ejemplo "
        "[politicas::1.1-1.3]. No uses números de sección ni inventes ids. Cada afirmación lleva su cita.\n"
        "4. Si el contexto no contiene la respuesta, dilo y ofrece derivar al cliente con un agente "
        "humano. No adivines.\n\n"
        "CONTEXTO:\n{contexto}\n\n"
        "PREGUNTA DEL CLIENTE: {pregunta}"
    ),
    # Variante para el experimento de la Fase 3: respuestas breves y sin conversación.
    "estricto": (
        "Eres un asistente de EcoMarket. Responde en español en máximo tres frases.\n"
        "Usa solo el CONTEXTO y cita los ids entre corchetes. Si el CONTEXTO no basta, responde "
        "exactamente: \"No tengo esa información; la derivo con un agente humano.\"\n\n"
        "CONTEXTO:\n{contexto}\n\n"
        "PREGUNTA: {pregunta}"
    ),
}


# ------------------------------------------------------------------------- recuperación
class Recuperador:
    """Pasos [1] y [2]: búsqueda por similitud + enlace FAQ -> política."""

    def __init__(self, almacen, k=K, enlace=True):
        self.almacen = almacen
        self.k = k
        self.enlace = enlace
        # El enlace es una consulta por id, no una búsqueda semántica. Se arma una sola vez: un
        # diccionario sección ('1.3') -> fragmento de políticas que la contiene. Las claves
        # vienen del metadato `claves` que indexar.py guardó ('1.1,1.2,1.3' para el fragmento
        # fusionado), por eso '1.1', '1.2' y '1.3' apuntan al mismo fragmento.
        previo = almacen.get(where={"fuente": "politicas"}, include=["documents", "metadatas"])
        self.por_clave = {}
        for id_, texto, meta in zip(previo["ids"], previo["documents"], previo["metadatas"]):
            doc = Document(id=id_, page_content=texto, metadata={**meta, "origen": "enlace"})
            for clave in filter(None, meta.get("claves", "").split(",")):
                self.por_clave[clave] = doc

    def recuperar(self, pregunta):
        # [1] RECUPERAR: top-k por similitud coseno. `score` es la DISTANCIA (menor = más cercano).
        pares = self.almacen.similarity_search_with_score(pregunta, k=self.k)
        docs = []
        for doc, distancia in pares:
            doc.metadata = {**doc.metadata, "origen": "similitud", "distancia": round(distancia, 4)}
            docs.append(doc)

        # [2] ENLAZAR (E): si una FAQ quedó entre los k primeros, se añade la sección de políticas
        # que la gobierna (metadato `politica_ref`). Va al final y sin repetir fragmentos. Es
        # determinista y no depende del modelo de embeddings: red de seguridad para los casos
        # en que la FAQ (pregunta con forma de pregunta de cliente) le gana a la política.
        if self.enlace:
            presentes = {d.id for d in docs}
            for doc in list(docs):
                if doc.metadata.get("fuente") != "faq":
                    continue
                for clave in filter(None, doc.metadata.get("politica_ref", "").split(",")):
                    vinculado = self.por_clave.get(clave)
                    if vinculado is not None and vinculado.id not in presentes:
                        docs.append(vinculado)
                        presentes.add(vinculado.id)
        return docs


def formatear_contexto(docs):
    """[3] ARMAR: cada fragmento con su id; el modelo cita ese id en la respuesta."""
    return "\n\n".join(f"[{d.id}]\n{d.page_content}" for d in docs)


# ------------------------------------------------------------------------- cadena LCEL
def crear_recuperador(ruta_db=RUTA_DB, k=K, enlace=True, embeddings=None):
    """Abre el índice Chroma y devuelve el Recuperador (pasos [1] y [2]). Se separa de
    crear_rag para poder evaluar la recuperación sin cargar el modelo generador."""
    if embeddings is None:
        from langchain_ollama import OllamaEmbeddings
        # bge-m3 no usa prefijos de consulta/documento (medido en la Fase 1).
        embeddings = OllamaEmbeddings(model=MODELO_EMBEDDINGS, base_url=OLLAMA)
    almacen = Chroma(collection_name=COLECCION, persist_directory=str(ruta_db),
                     embedding_function=embeddings)
    if almacen._collection.count() == 0:
        raise RuntimeError(f"El índice en {ruta_db} está vacío. Ejecuta primero: python indexar.py")
    return Recuperador(almacen, k=k, enlace=enlace)


def crear_rag(ruta_db=RUTA_DB, k=K, enlace=True, plantilla="base", modelo_llm=MODELO_LLM,
              embeddings=None, llm=None):
    """Devuelve una cadena que recibe la pregunta (str) y entrega un dict con
    'pregunta', 'docs' (fragmentos usados) y 'respuesta'. `embeddings` y `llm` se pueden
    inyectar (pruebas sin Ollama)."""
    if llm is None:
        from langchain_ollama import ChatOllama
        # temperature=0: respuestas repetibles, importante para comparar experimentos.
        # reasoning=False: qwen3 trae un modo "pensar" que antepone un bloque <think>; para
        # atención al cliente solo interesa la respuesta final (y ahorra segundos por consulta).
        # num_ctx=8192: el contexto por defecto de Ollama es corto; k fragmentos + prompt caben aquí.
        llm = ChatOllama(model=modelo_llm, base_url=OLLAMA, temperature=0,
                         reasoning=False, num_ctx=8192)

    recuperador = crear_recuperador(ruta_db, k, enlace, embeddings)
    prompt = ChatPromptTemplate.from_template(PROMPTS[plantilla])

    generar = (RunnableLambda(lambda x: {"contexto": formatear_contexto(x["docs"]),
                                         "pregunta": x["pregunta"]})
               | prompt | llm | StrOutputParser())          # [3] ARMAR -> [4] GENERAR

    # RunnableParallel ejecuta [1]+[2] y conserva la pregunta; .assign añade la respuesta
    # sin perder los fragmentos, que se necesitan para citar y auditar.
    cadena = (RunnableParallel(docs=RunnableLambda(recuperador.recuperar),
                               pregunta=RunnablePassthrough())
              .assign(respuesta=generar))
    return cadena


def citas_invalidas(respuesta, docs):
    """Ids citados entre corchetes que no corresponden a ningún fragmento entregado al modelo.
    El modelo puede parafrasear el id (p. ej. '1.3' en lugar de '1.1-1.3'); la verificación es
    determinista y permite medir cuántas veces ocurre."""
    citados = set(re.findall(r"\[([a-z]+::[^\]\s]+)\]", respuesta))
    return sorted(citados - {d.id for d in docs})


def imprimir(resultado, mostrar_contexto):
    if mostrar_contexto:
        print("\n--- Fragmentos entregados al modelo ---")
        for d in resultado["docs"]:
            extra = f"distancia {d.metadata['distancia']}" if d.metadata["origen"] == "similitud" \
                else "añadido por enlace"
            print(f"[{d.id}] ({extra})")
            print("   " + d.page_content.replace("\n", "\n   ")[:300] + "...\n")
    print("--- Respuesta ---")
    print(resultado["respuesta"].strip())
    malas = citas_invalidas(resultado["respuesta"], resultado["docs"])
    if malas:
        print(f"\n[aviso] citas que no corresponden a un fragmento entregado: {malas}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pregunta", nargs="?", help="pregunta del cliente")
    ap.add_argument("--interactivo", action="store_true")
    ap.add_argument("--k", type=int, default=K, help=f"fragmentos por similitud (defecto {K})")
    ap.add_argument("--sin-enlace", action="store_true", help="apaga el enlace FAQ -> política")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="base")
    ap.add_argument("--llm", default=MODELO_LLM)
    ap.add_argument("--ruta-db", default=str(RUTA_DB))
    ap.add_argument("--mostrar-contexto", action="store_true")
    a = ap.parse_args()
    if not a.pregunta and not a.interactivo:
        ap.error("indica una pregunta o usa --interactivo")

    rag = crear_rag(Path(a.ruta_db), a.k, not a.sin_enlace, a.prompt, a.llm)
    if a.interactivo:
        while True:
            try:
                q = input("\nCliente> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q:
                imprimir(rag.invoke(q), a.mostrar_contexto)
    else:
        imprimir(rag.invoke(a.pregunta), a.mostrar_contexto)
