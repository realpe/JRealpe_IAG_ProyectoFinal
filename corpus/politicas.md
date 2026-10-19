# Políticas de servicio al cliente — EcoMarket

> Documento de conocimiento **semi-estático**. En la arquitectura propuesta (ver `fase1_seleccion_modelo.md`)
> este es exactamente el tipo de contenido que se indexa en un **RAG**: cambia pocas veces al mes,
> es versionable y debe poder citarse en la respuesta al cliente.
> Versión 3.1 — vigente desde 2026-08-01.

---

## 1. Política de devoluciones

### 1.1 Plazo general
El cliente tiene **30 días calendario** contados desde la **fecha de entrega efectiva** para solicitar
una devolución, siempre que el producto sea elegible según la sección 1.2.

### 1.2 Elegibilidad por categoría de producto

| Categoría | ¿Admite devolución? | Condiciones |
|---|---|---|
| `DURADERO` | **Sí** | Producto sin uso, con empaque original y etiquetas. Reembolso al medio de pago original en 5–10 días hábiles. |
| `HIGIENE_PERSONAL` | **Solo si el empaque está sin abrir** | Por normativa sanitaria, un producto de higiene con el sello roto no puede reingresar al inventario. |
| `PERECEDERO` | **No** | Alimentos y productos con fecha de vencimiento no se reciben de vuelta. |

### 1.3 Excepción por producto defectuoso o dañado en transporte
La restricción de la sección 1.2 **no aplica** cuando el producto llegó dañado, incompleto o defectuoso.
En ese caso, **cualquier categoría** —incluidos perecederos y productos de higiene— da derecho a
**reposición o reembolso total**, sin importar el estado del empaque.

Requisito: el cliente debe reportarlo dentro de las **72 horas** siguientes a la entrega y adjuntar
evidencia fotográfica. El costo del transporte de retorno lo asume EcoMarket.

### 1.4 Casos que NO puede resolver el asistente automático
Deben escalarse a un agente humano:

- Solicitudes fuera del plazo de 30 días (requieren aprobación discrecional).
- Reclamos por más de **$500.000 COP**.
- Cualquier mención de daño a la salud, reacción alérgica o intoxicación.
- Amenaza de acción legal, queja ante la Superintendencia de Industria y Comercio, o exposición en redes.
- Segunda devolución del mismo cliente en un período de 60 días.

---

## 2. Política de envíos

- **Preparación**: 1–2 días hábiles desde la confirmación del pago.
- **Tiempos de tránsito estimados**: 2–3 días hábiles en ciudades principales; 4–7 días hábiles en el resto del país.
- **Envío gratis** en pedidos superiores a $120.000 COP.
- Las fechas de entrega son **estimadas**, no comprometidas contractualmente. El asistente **nunca** debe
  prometer una fecha exacta ni garantizar una entrega.

### 2.1 Manejo de retrasos
Cuando un pedido está marcado como `RETRASADO`, el asistente debe:

1. Reconocer el inconveniente y ofrecer una disculpa breve y sincera (sin excesos).
2. Informar el motivo registrado, si existe.
3. Entregar la nueva fecha estimada y el enlace de rastreo.
4. Ofrecer las opciones vigentes: esperar, o cancelar con reembolso total si el pedido aún no ha sido entregado.

**Prohibido**: ofrecer compensaciones, cupones o descuentos por iniciativa propia. Eso lo autoriza un humano.

---

## 3. Reglas de comunicación

- Tono cercano, claro y respetuoso. Tratar al cliente de **usted**.
- Español de Colombia, sin tecnicismos logísticos innecesarios.
- Máximo 150 palabras por respuesta, salvo que el cliente pida detalle.
- **Nunca inventar** un número de guía, una fecha o un estado. Si el dato no está en el contexto entregado,
  se responde que no se dispone de esa información y se ofrece escalar el caso.
- No solicitar ni repetir datos sensibles del cliente (documento de identidad, datos de tarjeta, dirección
  completa) en el cuerpo de la respuesta.
- Cerrar siempre ofreciendo ayuda adicional.

---

## 4. Criterio de escalamiento a humano

El asistente escala cuando detecta: enojo sostenido o lenguaje ofensivo, un problema que las políticas
anteriores no cubren, una solicitud de excepción, o tres intercambios sin resolver la consulta.
Al escalar, entrega al agente humano un **resumen del caso** con el `id_pedido` y lo ya intentado.
