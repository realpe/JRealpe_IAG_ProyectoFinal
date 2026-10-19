# Fase 4 — Despliegue de la aplicación

## 1. Herramienta: Streamlit

Se eligió Streamlit frente a Gradio por tres razones. Tiene componentes de chat nativos (`st.chat_message`, `st.chat_input`) que resuelven el historial visual sin código adicional. Su estado de sesión (`st.session_state`) permite mantener el agente cargado, el historial de la conversación y el registro de auditoría entre mensajes sin recargar el índice ni el modelo. Y su barra lateral y sus desplegables permiten mostrar, al lado de cada respuesta, lo que el agente hizo para llegar a ella, que es lo que convierte la interfaz en una herramienta de observación y no solo de chat. Gradio habría bastado para un chat simple y tiene una interfaz de chat igual de rápida de montar; la diferencia está en el panel de operador.

## 2. Implementación

`app.py` tiene tres partes:

- **Chat.** Campo de texto, historial con burbujas de cliente y asistente, y el historial de mensajes LangChain que se pasa al agente para que mantenga el contexto entre turnos.
- **Panel por respuesta.** Un desplegable "Cómo se resolvió" con la ruta del router, cada herramienta llamada con sus argumentos y resultado, los fragmentos recuperados si fue por RAG, la decisión de política, la etiqueta emitida y, en los casos de riesgo, el resumen que se entrega al agente humano.
- **Barra lateral.** Selección de modelo (local, o de API si hay clave en el entorno), botones de nueva conversación y de reinicio del estado simulado, los pedidos de prueba con su caso, prompts de ejemplo de un clic, y las últimas llamadas del registro de auditoría.

El agente se crea una vez por sesión. Sin clave de API, la aplicación corre completa con el modelo local.

## 3. Ejecución

```bash
source .venv/bin/activate
python indexar.py          # solo la primera vez
streamlit run app.py
```

La aplicación abre en `http://localhost:8501`.

## 4. Demostración

El video recorre, con los prompts de ejemplo de la barra lateral:

1. Devolución elegible por producto dañado (EM-1001): consultar, verificar por la sección 1.3, etiqueta emitida; el panel muestra las tres llamadas.
2. Devolución no elegible (EM-1002, alimento): la explicación cita la sección 1.2 y no hay etiqueta.
3. Escalamiento por plazo (EM-1003) y por monto (EM-1005): el agente informa la derivación a humano.
4. Pregunta informativa (plazo de devolución): ruta RAG con los fragmentos citados.
5. Caso de riesgo (sarpullido): respuesta fija sin modelo y resumen para el agente humano.
6. Intento de inyección (etiqueta sin verificar): el agente se niega; la herramienta lo haría de todos modos.

Se incluye además el cambio de modelo en la barra lateral y el registro de auditoría creciendo con cada turno.
