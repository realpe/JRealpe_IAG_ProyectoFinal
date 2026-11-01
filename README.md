# EcoMarket · Agente de devoluciones — Proyecto Final (IA Generativa)

José Luis Realpe · Maestría en IA Aplicada, Universidad Icesi · entrega 1 de noviembre de 2026

Extiende el sistema RAG del [Taller 2](https://github.com/realpe/jrealpe_IAG_Tarea_02) con un agente que automatiza el proceso de devolución: consulta el pedido, verifica la elegibilidad según la política y genera la etiqueta de transporte, y deriva a un agente humano los casos que la política reserva para personas. Interfaz web en Streamlit.

| Fase | Documento |
|---|---|
| 1. Diseño de la arquitectura del agente | [fase1_arquitectura.md](fase1_arquitectura.md) |
| 2. Implementación y conexión de componentes | [fase2_implementacion.md](fase2_implementacion.md) |
| 3. Análisis crítico y propuestas de mejora | [fase3_analisis_critico.md](fase3_analisis_critico.md) |
| 4. Despliegue de la aplicación | [fase4_despliegue.md](fase4_despliegue.md) |

## Resumen

- **Router determinista** a la entrada: riesgo (salud, legal) se escala sin modelo; informativa va al RAG del Taller 2; acción va al agente.
- **Cuatro herramientas**: `consultar_pedido`, `verificar_elegibilidad`, `generar_etiqueta_devolucion` y `buscar_politicas` (el RAG como herramienta). Las reglas de la política (secciones 1.1 a 1.4) son código con una prueba por regla.
- **Controles que no dependen del modelo**: la etiqueta exige una autorización HMAC que solo emite la verificación; el reclamo se evalúa sobre el acumulado del pedido (antifraccionamiento); tope de etiquetas por pedido; guardarraíl contra guías inventadas; cita de la sección de política en cada decisión; registro de auditoría de cada llamada y cada turno.
- **Modelo**: `qwen3:14b` local por defecto; Claude Haiku 4.5 por configuración. Evaluación de comportamiento con 17 casos: 17/17 y 16/17.
- **LangChain 1.x**, bucle de agente explícito, Streamlit.

## Instalación

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
ollama pull bge-m3 && ollama pull qwen3:14b
python indexar.py                 # índice Chroma de la base de conocimiento (54 fragmentos)
```

## Uso

```bash
streamlit run app.py                                            # interfaz web
python agente.py "Quiero devolver el termo del pedido EM-1001, me llegó abollado."
python agente.py --interactivo
```

Para usar el modelo de API: `export ANTHROPIC_API_KEY=...` (solo en la terminal; nunca en archivos del repositorio) y `--modelo anthropic:claude-haiku-4-5-20251001`, o selecciónelo en la barra lateral de la aplicación.

## Pruebas y evaluación

```bash
python test_reglas.py && python test_herramientas.py && python test_agente.py   # 35 pruebas, sin modelo
python prueba_tool_calling.py                                                   # decisión de herramientas por modelo
python evaluar_agente.py                                                        # 18 casos de comportamiento
python evaluar_agente.py --modelo anthropic:claude-haiku-4-5-20251001
```

Los resultados quedan en `resultados_agente_<modelo>.json`.

## Estructura

```
ecomarket-agente/
├── app.py                      Interfaz Streamlit
├── agente.py                   Router + RAG + bucle de herramientas; registro por turno
├── router.py                   Clasificación por reglas: riesgo / acción / informativa
├── herramientas.py             Las cuatro herramientas y el registro de auditoría
├── reglas_devolucion.py        Política de devoluciones como código (secciones 1.1 a 1.4)
├── datos/pedidos.json          Pedidos simulados que cubren todos los casos
├── casos_agente.json           18 casos de comportamiento esperado
├── evaluar_agente.py           Evaluador del comportamiento
├── prueba_tool_calling.py      Prueba previa de decisión de herramientas
├── test_*.py                   Pruebas unitarias sin modelo
├── corpus/, base_conocimiento.py, indexar.py, rag_ecomarket.py,
│   medir_embeddings.py, medir_segmentacion.py       Base de conocimiento y RAG del Taller 2
└── fase1..4_*.md               Documentos de las cuatro fases
```

Los datos son simulados; el corpus es sintético y describe una empresa ficticia.
