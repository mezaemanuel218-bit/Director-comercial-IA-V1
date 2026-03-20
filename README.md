# Director Comercial IA

Asistente comercial para Flotimatics basado en:

- Datos reales de Zoho CRM
- Documentos PDF internos de productos y servicios
- Razonamiento con GPT sin inventar datos

El objetivo no es crear un chatbot general, sino un asistente del departamento comercial que sirva para dos casos:

- Operacion diaria de vendedores
- Auditoria y supervision comercial

## Objetivo del sistema

El asistente debe poder responder preguntas como:

- "Dame los correos de X empresa"
- "A quien le hable ayer"
- "Que compromisos pendientes tengo hoy"
- "Que cliente tiene mas de 30 dias sin contacto"
- "Cuantas interacciones tiene cada vendedor"
- "Compara este prospecto contra otro"
- "Dame un plan comercial para cerrar este cliente"

Siempre debe:

- Priorizar Zoho y los PDFs como fuentes principales
- Decir claramente cuando algo viene de la web
- Mostrar fuentes cuando use web
- Evitar inventar informacion no respaldada

## Modos de respuesta

### 1. Modo datos

Se usa cuando la pregunta pide precision y respuesta corta.

Ejemplos:

- correos
- telefonos
- ultimo contacto
- propietario del contacto
- clientes sin seguimiento
- interacciones por vendedor

Reglas:

- Responder solo con lo pedido
- No agregar estrategia ni relleno
- Si no hay evidencia, decirlo claramente

### 2. Modo analisis

Se usa cuando la pregunta pide interpretacion, comparativa o plan.

Ejemplos:

- estrategia comercial
- comparativa entre prospectos
- a quien debo llamar hoy y por que
- riesgos y oportunidades

Reglas:

- Basarse solo en evidencia recuperada
- Separar hechos de interpretacion
- Explicar por que recomienda algo

### 3. Modo mixto

Combina datos y analisis cuando la pregunta lo necesite.

Ejemplo:

- "Dame a quien debo llamar hoy, sus numeros y por que"

## Fuentes del sistema

### Zoho CRM

Fuente principal operacional:

- Leads
- Contacts
- Notes
- Calls
- Events
- Tasks
- Owner / propietario del contacto

### PDFs en `doc/`

Fuente principal de conocimiento comercial:

- productos
- servicios
- presentaciones
- propuestas
- argumentos comerciales
- comparativas tecnicas

### Web

Fuente opcional y controlada:

- solo cuando el usuario lo pida
- o cuando el asistente lo considere necesario y lo declare explicitamente
- siempre citando fuentes
- nunca mezclando web como si fuera CRM interno

## Arquitectura objetivo

El sistema final debe tener 5 capas:

1. Ingestion
   Sincroniza Zoho y extrae PDFs.

2. Normalizacion
   Convierte datos de Zoho a un esquema unico y consultable.

3. Recuperacion
   Usa SQL para preguntas estructuradas y RAG para notas/PDFs.

4. Orquestacion
   Decide si la pregunta va por modo datos, analisis o mixto.

5. Respuesta final
   GPT redacta usando solo evidencia recuperada.

## Ruta recomendada de implementacion

### Fase 1. Base confiable

- Unificar esquema de base de datos
- Separar codigo de datos generados
- Corregir rutas y dependencias
- Definir pipeline oficial de sync y build

Pipeline oficial actual:

- `scripts/sync_zoho.py`
- `scripts/build_warehouse.py`
- `scripts/index_documents.py`
- `scripts/refresh_pipeline.py`
- `scripts/answer_question.py "tu pregunta"`
- `uvicorn api.app:app --reload --host 127.0.0.1 --port 8010`

### Fase 2. Motor de consultas

- Consultas por lead/contact/owner
- KPIs comerciales
- seguimientos
- compromisos pendientes
- ultima interaccion
- clientes sin contacto

### Fase 3. RAG comercial

- Indexar notas CRM
- Indexar PDFs
- Recuperar evidencia relevante por pregunta

### Fase 4. App web

- API backend
- autenticacion y roles
- interfaz para direccion y vendedores
- historial de preguntas

## API local

Ya existe una API base para conectar una app web:

- `POST /login`
- `POST /logout`
- `GET /me`
- `GET /health`
- `POST /refresh`
- `POST /ask`
- `GET /`

Para levantarla localmente:

```powershell
uvicorn api.app:app --reload --host 127.0.0.1 --port 8010
```

Swagger quedara disponible en:

- `http://127.0.0.1:8010/docs`

La interfaz web quedara disponible en:

- `http://127.0.0.1:8010/`

Si la quieres compartir en tu red local, levántala así:

```powershell
uvicorn api.app:app --reload --host 0.0.0.0 --port 8010
```

Luego comparte tu IP local con el puerto `8010`, por ejemplo:

- `http://192.168.1.25:8010/`

## Usuarios actuales

- `evaldez` -> Eduardo Valdez
- `pmelin` -> Pablo Melin
- `emeza` -> Emmanuel Meza

Contraseña actual:

- `Flotimatics2026`

## Despliegue

El proyecto ya queda preparado para desplegarse como contenedor.

Construcción local:

```powershell
docker build -t director-comercial-ia .
docker run -p 8010:8010 --env APP_SECRET_KEY="cambia-esta-clave" director-comercial-ia
```

Para publicarlo en internet todavía falta elegir un hosting o servidor donde desplegar el contenedor y configurar variables reales como:

- `OPENAI_API_KEY`
- `ZOHO_CLIENT_ID`
- `ZOHO_CLIENT_SECRET`
- `ZOHO_REFRESH_TOKEN`
- `APP_SECRET_KEY`

### Recomendación actual

La mejor opción gratuita para este proyecto hoy es **Render**.

Por qué:

- sí ofrece web services gratis actualmente
- es más simple que Fly.io para este caso
- acepta despliegue con Docker
- te da URL pública HTTPS
- luego se puede escalar a plan pagado sin rehacer todo

Limitaciones importantes:

- el servicio gratis puede dormirse tras inactividad
- el almacenamiento local no debe considerarse persistente para producción
- el historial guardado localmente puede perderse tras reinicios o redeploys

En tu caso, para arrancar gratis y validar uso real, sigue siendo la opción más práctica.

### Cómo subirlo a Render

1. Sube este proyecto a GitHub.
2. Crea una cuenta en Render.
3. Crea un nuevo `Web Service`.
4. Conecta el repositorio.
5. Render detectará el `Dockerfile` o el `render.yaml`.
6. Configura estas variables de entorno en Render:

- `OPENAI_API_KEY`
- `ZOHO_CLIENT_ID`
- `ZOHO_CLIENT_SECRET`
- `ZOHO_REFRESH_TOKEN`
- `APP_SECRET_KEY`

7. Publica el servicio.

La app quedará con un link tipo:

- `https://director-comercial-ia.onrender.com`

### Nota sobre datos e historial

Como el plan gratis no es buena opción para persistencia local, lo recomendable en esta etapa es:

- usar Zoho como fuente principal
- reconstruir el warehouse al arrancar
- aceptar que el historial pueda reiniciarse

Si más adelante quieren algo más serio, el siguiente salto natural sería:

- base de datos externa persistente
- disco persistente o backend pagado

Ejemplo de body para `POST /ask`:

```json
{
  "question": "a quien debo llamar hoy y por que"
}
```

## Criterios de calidad

- Una sola verdad para la base de datos
- Trazabilidad de respuesta
- Sin invenciones
- Respuestas distintas segun la intencion de la pregunta
- Preparado para app web

## Estado actual

El proyecto ya tiene piezas utiles, pero aun no tiene una arquitectura unica. Existen varios flujos y esquemas coexistiendo. El siguiente paso es consolidar todo en un pipeline oficial.

La nueva base oficial ya incluye:

- `assistant_core/warehouse.py`
- `assistant_core/documents.py`
- `assistant_core/query_intent.py`
- `assistant_core/service.py`
