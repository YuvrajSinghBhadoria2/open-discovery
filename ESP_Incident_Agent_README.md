th # ESP Incident Agent

FastAPI-based incident automation service for Enterprise Service Platform (ESP) / ServiceNow incidents. The service listens to Kafka for new incident events, processes them through a **LangGraph** workflow with tool looping, calls an external AI agent for resolution content, and executes **update** and **resolve** actions against ServiceNow. A live dashboard UI streams incident activity in real time.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Service](#running-the-service)
- [Live Dashboard UI](#live-dashboard-ui)
- [API Reference](#api-reference)
- [LangGraph Workflow](#langgraph-workflow)
- [Incident Business Logic](#incident-business-logic)
- [Kafka Consumer](#kafka-consumer)
- [Tools](#tools)
- [Adding Tools and Remediations](#adding-tools-and-remediations)
- [Summary Metrics Push](#summary-metrics-push)
- [Logging & Monitoring](#logging--monitoring)
- [Manual Testing](#manual-testing)
- [Troubleshooting](#troubleshooting)
- [Legacy Compatibility](#legacy-compatibility)
- [Security Notes](#security-notes)
- [License](#license)

---

## Overview

This service replaces the standalone Kafka consumer + script-based incident handler with a unified FastAPI application:

| Component | Role |
|---|---|
| FastAPI (`main.py`) | Process entrypoint: HTTP API, lifespan, live UI (stays at project root) |
| Kafka Consumer (`consumer.py`) | Background thread; polls Kafka and POSTs to the incident API |
| LangGraph (`agent/graph.py`) | Orchestrates AI fetch → planning → tool execution loop |
| Tools (`agent/tools.py`) | LangChain tools wrapping the ESP client in `agent/esp/` |
| Controllers (`api/controller/`) | REST handlers, one module per resource |
| Services (`api/services/`) | In-memory incident / activity / approval stores + worker pool |
| MongoDB (`api/db_ops/`) | Incident metadata and live-status persistence |

### End-to-end flow

1. Kafka message arrives with key `incident` and a matching **service offering**. The record is always stored in the agent.
2. The action workflow is bound only when the **assignment group** is in `FILTER_ASSIGNMENT_GROUPS` and that group has a duty-engineer listing. Matching incidents are POSTed to `POST /api/v1/incidents/process`.
3. LangGraph workflow:
   - Calls external AI agent for resolution instructions.
   - Applies business rules (new vs reopened incident) using the duty-engineer listing for that assignment group.
   - Loops through `update_incident_tool` and/or `resolve_incident_tool`.
4. Each step is logged to the activity store and visible on the live dashboard.

---

## Architecture

```text
┌──────────────────┐   POST /api/v1/incidents/process   ┌──────────────────┐
│ Kafka Topic      │ ─────────────────────────────────▶ │ FastAPI Service  │
│ esp-ITSM-NRT-*   │      (consumer.py thread)          │ (main.py)        │
└──────────────────┘                                    └────────┬─────────┘
                                                                 │
                                                                 ▼
                                                        ┌──────────────────┐
                                                        │ LangGraph Workflow│
                                                        │ (agent/graph.py) │
                                                        └────────┬─────────┘
                                                                 │
                         ┌───────────────────────────────────────┼───────────────────────────┐
                         │                                       │                           │
                         ▼                                       ▼                           ▼
                ┌──────────────────┐                  ┌──────────────────────┐    ┌───────────────────┐
                │ External AI Agent│                  │ update_incident_tool │    │ resolve_incident  │
                │ (edag-ai-agent)  │                  │ (agent/esp/update)   │    │ _tool (esp/)      │
                └────────┬─────────┘                  └──────────┬───────────┘    └─────────┬─────────┘
                         │                                       │                          │
                         └───────────────────────────────────────┼──────────────────────────┘
                                                                 │
                                                                 ▼
                                                        ┌──────────────────┐
                                                        │ Activity Store   │ ──▶ SSE ──▶ Live UI
                                                        │ + MongoDB metadata│
                                                        └──────────────────┘
```

---

## Project Structure

`main.py` stays at the project root (process entrypoint). HTTP, stores, and Mongo live under `api/`. LangGraph and the ServiceNow client live under `agent/`.

```text
esp-ops-ai-agent/
├── main.py                      # FastAPI entry; lifespan starts Kafka + worker pool
├── consumer.py                  # Background Kafka consumer thread
├── call_incident_agent.py       # Legacy wrapper + delegates to LangGraph
├── call_dutypager.py            # Duty roster / on-call engineer lookup
├── service_object.py            # Shared config and logger
├── push_esp_agent_summary_metrics.py  # Cron: daily AI case counts → S3 External Team Metrics API
├── config.ini                   # Application secrets and settings (do not commit real secrets)
├── requirements.txt
├── README.md
│
├── agent/
│   ├── assignment_scope.py      # Service-offering intake vs assignment-group action
│   ├── graph.py                 # LangGraph state machine and workflow
│   ├── tools.py                 # LangChain tools wrapping agent/esp
│   ├── tool_registry.py         # Pluggable tool registry
│   ├── workflow_registry.py     # Incident type → tool loop
│   ├── approval_handler.py      # Human approval for remediations
│   ├── ai_planner.py
│   ├── mcp_client.py
│   ├── platform.py
│   ├── remediation.py
│   ├── esp/                     # ServiceNow / ESP HTTP client
│   │   ├── client.py            # Shared OAuth token (from config.ini)
│   │   ├── update.py            # Incident update (was update_incident.py)
│   │   └── resolve.py           # Incident resolve (was resolve_incident.py)
│   └── remediation_plugins/     # Platform remediations (MCP), not ticket update/resolve
│       ├── base.py
│       └── builtin.py
│
├── api/
│   ├── schemas.py               # Pydantic request/response models
│   ├── controller/              # REST handlers (mounted by main.py from api.controller)
│   │   ├── health.py            # GET /api/v1/health
│   │   ├── incidents.py         # list / process / get incident
│   │   ├── workflows.py         # GET /api/v1/workflows
│   │   ├── tools.py             # GET /api/v1/tools
│   │   ├── approvals.py         # list / approve / reject
│   │   ├── activities.py        # list + SSE stream
│   │   └── consumer.py          # GET /api/v1/consumer/status
│   ├── services/                # In-memory stores + process pool
│   │   ├── activity_store.py    # Live activity events + SSE
│   │   ├── approval_store.py    # Pending admin approvals
│   │   ├── incident_registry.py # Kafka incident registry
│   │   └── incident_worker_pool.py
│   └── db_ops/                  # MongoDB operations
│       ├── client.py            # Mongo connection (from config.ini)
│       ├── incidents.py         # INCIDENT_AGENT_METADATA
│       └── live_status.py       # Live UI state across restarts
│
├── static/
│   └── index.html               # Live incident dashboard
│
├── logs/
│   └── incident_agent.log
│
└── CyberArk_Cert/
    └── Users_Safe/              # mTLS certs for CyberArk (keep private key out of git)
        ├── espmsgdev_cert_2025-26.pem
        └── espmsgdev_cert_2025-26.key
```

### Layout notes

| Path | Why it is there |
|---|---|
| `main.py` (root) | Service bootstrap only — not an HTTP handler |
| `api/controller/` | One module per REST resource; `api/routes.py` was removed |
| `api/services/` | In-memory stores used by controllers and the agent |
| `api/db_ops/` | All Mongo reads/writes (was root `db_ops.py` + `services/live_status_store.py`) |
| `agent/esp/` | Outbound ServiceNow client used by LangGraph tools |

---

## Prerequisites

- Python 3.11+
- Network access to:
  - Kafka cluster (dev/stage/prod bootstrap servers)
  - CyberArk CCP (for Kafka SASL password)
  - External AI agent API (`edag-ai-agent`)
  - ServiceNow ESP (`ciscodev.service-now.com`)
  - MongoDB cluster
  - Duty pager API (for on-call engineer assignment)
- CyberArk certificates in `CyberArk_Cert/Users_Safe/`
- Virtual environment (recommended)

---

## Installation

```bash
cd esp-ops-ai-agent

# Create and activate virtual environment (optional)
python -m venv .espvenv
source .espvenv/bin/activate   # Linux/macOS
# .espvenv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `fastapi`, `uvicorn` | HTTP API server |
| `langgraph`, `langchain-core` | Agent workflow and tools |
| `confluent-kafka` | Kafka consumer |
| `pymongo`, `certifi` | MongoDB with TLS |
| `requests` | HTTP clients (AI agent, ServiceNow, duty pager) |
| `pydantic` | Request/response validation |

---

## Configuration

### `config.ini`

Copy and edit `config.ini` with your environment values. **Do not commit secrets to version control.**

```ini
[AGENT_API_CONFIG]
ai_agent_api_url = http://<ai-agent-host>:5000/agent/chat-with-incident
ai_agent_api_token = <your-ai-agent-bearer-token>

[APP]
host = 0.0.0.0
port = 8080
base_url = http://127.0.0.1:8080

[MONGO_DETAILS]
connection_string = mongodb://<user>:<password>@<host>:<port>/<database>?ssl=true
mongo_database = <database_name>

[LOGGING]
BASE_LOGGER = incident_agent
LOG_FORMAT = %(asctime)s : loglevel = %(levelname)s : msg = %(message)s : line = %(lineno)d : codeblock

[ESP_INCIDENT]
esp_url = https://ciscodev.service-now.com
client_id = <service-now-oauth-client-id>
client_secret = <service-now-oauth-client-secret>
user = data_plat_agent.gen
password = <optional>

[DUTY_PAGER]
client_id = <duty-pager-oauth-client-id>
client_secret = <duty-pager-oauth-client-secret>
```

### `config.ini` `[KAFKA_CONSUMER]` — Kafka and CyberArk

Kafka broker, topic, SASL, worker pool, and CyberArk mTLS settings are read from this section. The Kafka password is fetched from CyberArk at runtime — do not put it in `config.ini`.

| Parameter | Description |
|---|---|
| `process_pool_size` | Parallel incident workers; `0` = CPU core count |
| `bootstrap_server` | Kafka broker list |
| `consumer_group_Id` | Consumer group ID |
| `topic_to_consume` | Incident topic name |
| `sasl_username` | Kafka SASL username |
| `security_protocol` | e.g. `SASL_PLAINTEXT` |
| `sasl_mechanisms` | e.g. `SCRAM-SHA-512` |
| `auto_offset_reset` | `latest` or `earliest` |
| `session_timeout_ms` | Kafka session timeout |
| `enable_auto_offset_store` | Whether the client stores offsets automatically |
| `compression_type` | e.g. `snappy` |
| `app_id`, `obj_name` | CyberArk safe object for Kafka password |
| `secret_timeout` | CyberArk HTTP timeout (seconds) |
| `client_cert`, `client_key` | mTLS cert paths for CyberArk |

### Environment variables

| Variable | Description | Default |
|---|---|---|
| `ESP_AGENT_HOST` | Bind host | `config.ini` `[APP]` `host` |
| `ESP_AGENT_PORT` | Bind port | `config.ini` `[APP]` `port` |
| `ESP_AGENT_BASE_URL` | URL consumer uses to call API | `config.ini` `[APP]` `base_url` |

---

## Running the Service

```bash
cd esp-ops-ai-agent
source ../.espvenv/bin/activate   # if using venv
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8080
```

On startup:

1. FastAPI application starts.
2. Kafka consumer thread starts automatically (via lifespan hook).
3. Live dashboard is available at <http://localhost:8080>.

Interactive API docs: <http://localhost:8080/docs>

---

## Live Dashboard UI

Open <http://localhost:8080> in a browser.

The dashboard shows:

- **System status** — Kafka consumer running/stopped, last poll time, messages processed
- **Live activity feed** — Real-time incident events via Server-Sent Events (SSE)
- **Event details** — Incident number, action type, status badge, timestamp, message

### Event status badges

| Status | Meaning |
|---|---|
| `info` | Informational event |
| `in_progress` | Action in progress |
| `success` | Completed successfully |
| `error` | Failed |
| `warning` | Completed with warnings |
| `skipped` | Incident skipped by business rules |

### SSE endpoint

The UI connects to `GET /api/v1/activities/stream` for live updates. On reconnect, it replays the last 50 events.

---

## API Reference

Base path: `/api/v1`. Handlers live in `api/controller/` and are included from `main.py` via `api.controller.router`.

### `GET /`

Serves the live dashboard HTML (`static/index.html`).

### `GET /api/v1/health`

Health check.

**Response:**

```json
{
  "status": "ok",
  "consumer_running": true,
  "kafka_connected": true
}
```

### `POST /api/v1/incidents/process`

Process a single incident through the LangGraph workflow.

**Request body:**

```json
{
  "incident_object": {
    "requestor": "User Name (userid)",
    "description": "Incident description text",
    "number": "INC12345678",
    "state": "Awaiting Assignment",
    "service_offering": "Data Integration Platform Services",
    "assignment_group": "EDS-PLATFORM-SNOWFLAKE",
    "assigned_to": "",
    "urgency": "3 - Low",
    "priority": "5 - Standard",
    "category": "Access",
    "subcategory": "Authentication",
    "opened_at": "2026-06-11 05:02:34",
    "last_updated_at": "2026-06-11 05:14:19",
    "reopen_count": "0"
  },
  "source": "kafka"
}
```

`source` values: `kafka` | `api` | `manual`

**Response:**

```json
{
  "success": true,
  "incident_number": "INC12345678",
  "status": "completed",
  "actions": [
    {
      "tool": "update_incident_tool",
      "result": "{\"success\": true, \"result\": {...}}"
    }
    ...
  ],
  "message": "Workflow finished with status: completed"
}
```

> Note: the middle part of this response example (between the first action entry and the `message` field) was not visible in the source photos.

### `GET /api/v1/incidents/{incident_number}`

Returns MongoDB metadata and activity events for one incident.

**Response:**

```json
{
  "metadata": { "INCIDENT_NUMBER": "INC12345678", "STATE": "Resolved", ... },
  "activity": [ { "id": "...", "action": "update_incident", ... } ]
}
```

### `GET /api/v1/activities?limit=100`

Returns recent activity events (newest first).

### `GET /api/v1/activities/stream`

Server-Sent Events stream of live activity. Content-Type: `text/event-stream`.

### `GET /api/v1/consumer/status`

Kafka consumer runtime status.

**Response:**

```json
{
  "running": true,
  "last_poll": "2026-08-22 19:03:22 UTC",
  "messages_processed": 5,
  "last_error": null
}
```

---

## LangGraph Workflow

The incident processor is implemented as a LangGraph `StateGraph` in `agent/graph.py`.

### State (`IncidentAgentState`)

| Field | Description |
|---|---|
| `incident_object` | Full incident payload |
| `ai_content` | Resolution text from external AI agent |
| `messages` | LangChain messages for tool execution |
| `pending_tool_calls` | Queue of tools still to execute |
| `actions_taken` | Completed tool results |
| `status` | Current workflow status |
| `success` | Overall success flag |
| `skip_tools` | Skip tool execution (e.g. already handled) |

### Graph nodes

```text
fetch_ai → prepare_metadata → plan_tools
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
            dispatch_tool                    finalize → END
                  │
                  ▼
        execute_tools (ToolNode)
                  │
        ┌─────────┴──────────────────┐
        ▼                            ▼
  dispatch_tool              record_results → finalize → END
  (more tools?)                    (all tools done)
```

| Node | Description |
|---|---|
| `fetch_ai` | POST to external AI agent; stores resolution content |
| `prepare_metadata` | Load or store incident metadata in MongoDB |
| `plan_tools` | Apply business rules; queue tool calls |
| `dispatch_tool` | Dequeue next tool call into an `AIMessage` |
| `execute_tools` | LangGraph `ToolNode` runs the tool |
| `record_results` | Collect `ToolMessage` results |
| `finalize` | Set final status and log to activity store |

### Tool looping

For new incidents, the planner queues **two tools** in sequence (document / Q&A resolution, **no admin approval**):

1. `update_incident_tool` — add AI comments and assign to `data_plat_agent.gen`
2. `resolve_incident_tool` — resolve the incident with resolution notes

The graph loops: `dispatch_tool → execute_tools → dispatch_tool → ...` until the queue is empty. If a later workflow inserts a critical remediation with `requires_approval=True`, `dispatch_tool` pauses before `ToolNode` until an admin approves.

---

## Incident Business Logic

Logic in `plan_tools` (mirrors original `call_incident_agent.py` behavior).

### Approval policy

| Kind of work | Tools | Admin checkpoint |
|---|---|---|
| Document / Q&A resolution (current) | `update_incident_tool` → `resolve_incident_tool` | **None** — runs automatically |
| Critical remediation (later) | grant access, restart job, MCP execute, etc. | **Yes** — set `requires_approval=True` on that tool only |

Do not gate update/resolve. Approval is only for execute-on-platform remediations when those steps are added to a workflow.

### Current workflows (`agent/workflow_registry.py`)

All active paths are Q&A-style (no approval). Remediation `ToolStep`s are commented in the registry until those actions are ready.

| Workflow | Incident type | Sequence | Approval |
|---|---|---|---|
| `standard_self_service` | `general_inquiry` | update → resolve | No |
| `access_request` | `access_request` | update → resolve | No (add gated grant later) |
| `job_failure` | `job_failure` | update → resolve | No (add gated job restart later) |
| `reopened_assign` | `reopened` | update (assign engineer) | No |

When a critical remediation is ready, insert it before resolve, for example:

```python
ToolStep("update_incident_tool", _update_args, requires_approval=False),
ToolStep("access_grant_remediation", _remediation_args, requires_approval=True),
ToolStep("resolve_incident_tool", _resolve_args, requires_approval=False),
```

The graph already stops before `ToolNode` for any name with `requires_approval=True` and shows **why / what** (`trigger`, `action_summary`, `command`) on `GET /api/v1/approvals`.

### Skip conditions

- Incident state is `Closed`, `Resolved`, or `Pending`
- Incident already assigned to a human engineer (not empty, not `data_plat_agent.gen`)
- Assignment group is not in `FILTER_ASSIGNMENT_GROUPS` (record is still stored; `workflow_status=stored`)
- Assignment group has no duty-engineer listing in `DUTY_ROSTERS`

### Reopened incidents (`reopen_count > 0`)

| Condition | Action |
|---|---|
| Previously resolved, not yet marked reopened | Assign to duty engineer via `update_incident_tool` with `reopened=True` |
| Reopened but no metadata | Store metadata, assign to duty engineer |
| Already reopened and assigned | Skip — no further action |

Duty engineer is selected from:

- Static engineer list per assignment group, or
- Live duty pager API (`call_dutypager.py`)

`EXEMPTED_DUTY_USERS` are removed from **both** lists before a selection is made. Case-insensitive CEC match. If every remaining user is exempted and there is no other eligible engineer, assignment fails.

### New incidents (`reopen_count == 0`)

| Condition | Action |
|---|---|
| Unassigned or assigned to `data_plat_agent.gen` | `update_incident_tool` → `resolve_incident_tool` with AI content |
| Assigned to human engineer | Skip |

### Platform mapping

Assignment group is mapped to an AI agent platform (e.g. `EDS-PLATFORM-SNOWFLAKE` → `data-ingestion`) before calling the external AI API.

Action workflow binding uses `FILTER_ASSIGNMENT_GROUPS` first, then the duty-engineer listing (`DUTY_ROSTERS` / duty pager) for that group. Groups outside the allow-list, or groups with no duty listing, are stored only.

When you add a group to `FILTER_ASSIGNMENT_GROUPS`, also add it to a `DUTY_ROSTERS` key (`agent/assignment_scope.py`) and to the mapping in `agent/platform.py`. On startup the service logs a `config_warning` (in `logs/incident_agent.log` and the SYSTEM activity feed) for every allow-listed group that is missing either one. Config changes only take effect after a service restart.

---

## Kafka Consumer

`consumer.py` runs as a **daemon background thread** started in the FastAPI lifespan.

### Message filtering

Lists live in `agent/assignment_scope.py`.

1. Message key must be `"incident"`
2. `u_business_service` header must match an allowed **service offering** (`SERVICE_OFFERINGS`). Matching incidents are **stored** in the agent registry.
3. Action workflow is bound only when `assignment_group` is in `FILTER_ASSIGNMENT_GROUPS` and that group has a duty-engineer listing (`DUTY_ROSTERS`). Other stored incidents stay at `workflow_status=stored`.
4. Incident state must not be `Closed` or `Resolved` for the action workflow to run

### Authentication

- Kafka SASL password retrieved from **CyberArk CCP** at runtime via mTLS using `[KAFKA_CONSUMER]` `client_cert` / `client_key`.
- Do not hardcode the Kafka password.

### Internal API call

After parsing, the consumer POSTs to:

```text
{ESP_AGENT_BASE_URL}/api/v1/incidents/process
```

Default: `http://127.0.0.1:8080/api/v1/incidents/process`

Ensure `base_url` in `config.ini` matches the port the FastAPI service listens on.

---

## Tools

Defined in `agent/tools.py` using the LangChain `@tool` decorator. The tools call the ServiceNow client in `agent/esp/` (`update.py`, `resolve.py`). Shared OAuth is in `agent/esp/client.py` and reads `[ESP_INCIDENT]` from `config.ini`.

### `update_incident_tool`

Updates a ServiceNow incident with agent comments and assignment.

| Parameter | Type | Description |
|---|---|---|
| `incident_number` | str | ESP incident number |
| `case_comments` | str | Main resolution/update text |
| `additional_comments` | str | Prefix instructions for requester |
| `user_name` | str | Assignee CEC ID |
| `state` | str | Current incident state |
| *(not visible in source photos)* | | |
| `reopened` | bool | Mark as reopened in metadata |

> Note: one or more parameter rows between `state` and `reopened` were not visible in the source photos.

### `resolve_incident_tool`

Resolves a ServiceNow incident.

| Parameter | Type | Description |
|---|---|---|
| `incident_number` | str | ESP incident number |
| `case_comments` | str | Resolution notes |
| `user_name` | str | Resolver CEC ID |
| `state` | str | Target state (default `Resolved`) |

Both tools log progress to `api/services/activity_store.py` and update MongoDB metadata via `api/db_ops/incidents.py`.

**Approval policy (current):** document / Q&A resolution does **not** use an admin checkpoint. `update_incident_tool` and `resolve_incident_tool` run automatically for general inquiry, access, and job-failure tickets. When a later workflow adds a **critical remediation** (grant access, restart a job, MCP execute), set `requires_approval=True` on that tool only. The graph already pauses before `ToolNode` for those names.

---

## Adding Tools and Remediations

The LangGraph graph does **not** need to change for each new tool. Plug-and-play is three layers:

```text
incident fields  →  workflow (tool sequence)  →  named plugin (execute)
classify_incident_type    workflow_registry          remediation_plugins
```

`agent/graph.py` only runs names that are already in the registry. Add a plugin file and a workflow mapping; do not edit graph nodes.

| Layer | File | What you add |
|---|---|---|
| Plugin (action) | `agent/remediation_plugins/` | One class per remediation (MCP or local) |
| Workflow (sequence) | `agent/workflow_registry.py` | Incident type → ordered tool names |
| Hint (optional) | `agent/graph.py` `_default_remediation_hint` | Swap generic `remediation_tool` for a plugin name |
| Core ESP tools | `agent/esp/` + `agent/tools.py` | ServiceNow update/resolve only — not remediations |

Do **not** put ServiceNow update/resolve inside `remediation_plugins/`. Those stay in `agent/esp/`.

### 1. Add a remediation plugin

Create a new module under `agent/remediation_plugins/` (any file except `base.py` / `builtin.py` is auto-discovered on startup). Example:

```python
# agent/remediation_plugins/snowflake_grant.py
from agent.mcp_client import call_mcp_tool
from agent.remediation_plugins.base import RemediationPlugin

class SnowflakeGrantPlugin(RemediationPlugin):
    name = "snowflake_grant_role"          # becomes the LangGraph tool name
    description = "Grant Snowflake role via MCP"
    mcp_server = "edag-platform"           # MCP gateway server id
    mcp_tool_name = "grant_role"           # tool on that server
    platforms = ("snowflake",)

    def build_plan(self, incident, ai_content, ctx, tool_args):
        return {
            "tool_name": self.name,
            "trigger": "Snowflake role grant",
            "action_summary": "Grant role to requestor",
            "command": f"mcp grant_role --incident {incident['number']}",
            "parameters": [
                "incident_number": incident["number"],
                ...
                "platform": ctx.get("platform"),
                "plugin_name": self.name,
            ],
            "incident_description": incident.get("description", "")[:500],
            "ai_guidance_preview": (ai_content or "")[:500],
        }

    def execute(self, tool_args):
        plan = tool_args.get("remediation_plan") or {}
        return call_mcp_tool(
            server=self.mcp_server,
            tool_name=self.mcp_tool_name,
            arguments=self.mcp_arguments(tool_args, plan),
        )
```

> Note: this code sample is transcribed from photographs that cut off between pages; the `parameters` block in `build_plan` was only partially visible (`"incident_number": incident["number"],` … `"platform": ctx.get("platform"), "plugin_name": self.name,`), so lines between them may be missing, and the original uses its own bracket/brace structure there.

Then either:

- add the class to `BUILTIN_PLUGINS` in `agent/remediation_plugins/builtin.py`, or
- leave the new file in the folder — `load_remediation_plugins()` imports extra modules automatically.

That registration creates a LangChain tool, requires admin approval, and runs MCP (or local logic) after approve.

Restart the service and confirm:

```bash
curl http://localhost:8080/api/v1/tools
```

MCP credentials belong in `config.ini` / the gateway — never hardcode tokens in the plugin.

### 2. Map incident type → workflow

In `agent/workflow_registry.py`:

**Classify** from category / subcategory / reopen:

```python
if "snowflake" in subcategory and category == "access":
    return "snowflake_access"
```

**Point** that type at a workflow id in `get_workflow_for_incident()`.

**Define** the loop as names only (no LangGraph code). Today keep Q&A without approval:

```python
"access_request": WorkflowDefinition(
    workflow_id="access_request",
    incident_type="access_request",
    label="Access Request",
    description="AI guidance + resolve. No approval until a critical remediation is added.",
    tools=[
        ToolStep("update_incident_tool", _update_args, requires_approval=False),
        ToolStep("resolve_incident_tool", _resolve_args, requires_approval=False),
    ],
),
```

**Later**, insert a gated remediation before resolve (checkpoint with why/what for the admin):

```python
"snowflake_access": WorkflowDefinition(
    workflow_id="snowflake_access",
    incident_type="snowflake_access",
    label="Snowflake Access",
    description="Comment, grant role (approval), then resolve.",
    tools=[
        ToolStep("update_incident_tool", _update_args, requires_approval=False),
        ToolStep("snowflake_grant_role", _remediation_args, requires_approval=True),
        ToolStep("resolve_incident_tool", _resolve_args, requires_approval=False),
    ],
),
```

If a step is still named `remediation_tool`, the graph swaps it for a hint. Optional default in `agent/graph.py`:

```python
def _default_remediation_hint(incident_type: str) -> str:
    return {
        "access_request": "access_grant_remediation",
        "job_failure": "job_restart_remediation",
        "snowflake_access": "snowflake_grant_role",
    }.get(incident_type, "")
```

Putting the plugin name directly on the `ToolStep` is clearer than `remediation_tool` + hint.

Confirm:

```bash
curl http://localhost:8080/api/v1/workflows
```

### 3. Optional: send tool discovery to the remote LLM

The LLM service does **not** store tool names. On each `fetch_ai` call this agent sends a **platform-filtered catalog** plus the default workflow. Plugin `platforms` must match `get_platform_from_assignment_group()` (for example `snowflake`, `informatica`). Core tools with an empty `platforms` tuple are sent for every incident.

**Request extras** (added next to the incident description):

```json
{
  "platform": "informatica",
  "available_tools": [
    {"name": "update_incident_tool", "category": "core", "requires_approval": false, "platforms": []},
    {"name": "informatica_create_directory", "category": "remediation", "requires_approval": true, "plat...
  ],
  "default_workflow": {
    "workflow_id": "access_request",
    "incident_type": "access_request",
    "tools": ["update_incident_tool", "resolve_incident_tool"]
  },
  "execution": {
    "executor": "esp-ops-ai-agent",
    "runs_tool_sequence": true,
    "unknown_names_skipped": true,
    "empty_sequence_uses_default_workflow": true,
    "instruction": "You do not execute tools. ... This ESP agent will validate and execute that list."
  }
}
```

> Note: the `available_tools` second entry and the `instruction` string were truncated (`...`) in the source photos.

The LLM does **not run** the sequence. It only returns names from `available_tools`. This service then:

1. Validates each name against `tool_registry` (unknown names dropped).
2. Executes the remaining list in order via LangGraph (`plan_tools → dispatch_tool → execute_tools`).
3. If `tool_sequence` is missing or empty, runs `default_workflow.tools` instead.

**Expected LLM response:**

```json
{
  "content": "Grant SYSADMIN to the user",
  "tool_sequence": ["update_incident_tool", "informatica_create_directory", "resolve_incident_tool"],
  "remediation_tool": "informatica_create_directory"
}
```

The remote `chat-with-incident` service must read `available_tools` / `execution` and echo `tool_sequence`. Until it does, omit the sequence and local mapping still runs.

### 4. What not to do

| Avoid | Why |
|---|---|
| Editing `graph.py` nodes for each tool | The loop already dispatches by name |
| Putting update/resolve in `remediation_plugins/` | Those stay in `agent/esp/` as core tools |
| Hardcoding MCP tokens or passwords in the plugin | Use `config.ini` / gateway auth |
| One giant `if category == ...` inside `execute` | One plugin class per action |

### Checklist

1. New class in `agent/remediation_plugins/` (`name`, `build_plan`, `execute`).
2. Map category/subcategory → workflow in `workflow_registry.py`.
3. Put the plugin name in that workflow's tool list (and optionally `_default_remediation_hint`).
4. Restart; confirm `GET /api/v1/tools` and `GET /api/v1/workflows`.

---

## Summary Metrics Push

`push_esp_agent_summary_metrics.py` is a standalone script (run once a day by cron, not by the FastAPI service). It pushes **one day** of AI case counts per assignment group to the DB team's **S3 External Team Metrics API** (`POST .../s3/external-metrics`), which feeds the Case Summary dashboard.

Each run:

1. Picks the metric date — by default yesterday in Pacific time (the last complete day).
2. Reads that day's counts per assignment group from Snowflake `ESP_AGENT_METRIC` (incidents opened that day, `OPENED_AT` in Pacific time).
3. Builds one metric row per assignment group + category. Groups without a sys_id in `ASSIGNMENT_GROUP_SYS_IDS` (top of the script) are skipped with a warning.
4. Gets a Duo OAuth access token (client credentials).
5. POSTs the payload in one request and logs accepted / rejected rows.

| Category | Meaning |
|---|---|
| `AI_RESOLVED` | Assigned to the agent (`data_plat_agent.gen`), AI responded, state Resolved/Closed |
| `REOPENED` | AI responded, then the incident was reopened and assigned to L2 |
| `MANUAL_QUEUE` | Assigned to L2 and not reopened — directly, or after an AI response |
| `AI_ADDRESSED` | `AI_RESOLVED` + `REOPENED` |

**Upsert:** the API upserts each row by team + `metric_date` + category + `assigned_group`, so running the same day again simply overwrites that day's counts (safe to re-run a failed day with `--date`).

| `config.ini` section | Keys |
|---|---|
| `[SNOWFLAKE_DETAILS]` | `account`, `user`, `password`, `role`, `warehouse`, `database`, `schema`, `table` (or `sf_*` env vars) |
| `[DUO]` | `Client ID`, `Client Secret`, `Discovery Url` (or `duo_client_id`, `duo_client_secret`, `duo_discovery_url` env vars) |
| `[S3_METRICS_API]` | `url`, `team_name` (`Data Platforms`), `scope` (optional) |

```bash
python push_esp_agent_summary_metrics.py                    # push yesterday (Pacific time)
python push_esp_agent_summary_metrics.py --date today       # push today's counts so far
python push_esp_agent_summary_metrics.py --date 2026-10-14  # push / re-push one specific day
python push_esp_agent_summary_metrics.py --dry-run          # print the payload, push nothing
```

Exit code is `0` when every row was accepted and `1` otherwise. Run it once a day after the Pacific-time day has ended and after `push_esp_agent_data_to_sf.py` has loaded Snowflake, for example at 14:00 IST:

```text
0 14 * * * cd /path/to/esp-ops-ai-agent && .venv/bin/python push_esp_agent_summary_metrics.py
```

**Pushing more often:** no code change is needed — add a cron entry with `--date today` (each push overwrites today's counts, so there are no duplicates) and keep the daily run above, which finalizes yesterday (incidents opened in the last minutes before Pacific midnight are only picked up by it). The numbers are only as fresh as the Snowflake load, so run `push_esp_agent_data_to_sf.py` at least as often, before the push. For example, every 15 minutes:

```text
*/15 * * * * cd /path/to/esp-ops-ai-agent && .venv/bin/python push_esp_agent_summary_metrics.py --date t...
0 14 * * *   cd /path/to/esp-ops-ai-agent && .venv/bin/python push_esp_agent_summary_metrics.py >> logs/...
```

> Note: both cron lines in this last example were cut off at the right edge in the source photo.

---

## Logging & Monitoring

### File log

```text
logs/incident_agent.log
```

Configured in `service_object.py` using `[LOGGING]` section from `config.ini`.

### Activity store

In-memory ring buffer (500 events) with SSE broadcast (`api/services/activity_store.py`). Live incidents, activities, and approvals are also persisted in MongoDB via `api/db_ops/live_status.py` and restored on startup.

### MongoDB collections

| Collection | Module | Purpose |
|---|---|---|
| `INCIDENT_AGENT_METADATA` | `api/db_ops/incidents.py` | Incident fields, agent responses, assignment, reopen flag |
| `INCIDENT_AGENT_LIVE_INCIDENTS` | `api/db_ops/live_status.py` | Live dashboard incident list |
| `INCIDENT_AGENT_LIVE_ACTIVITY` | `api/db_ops/live_status.py` | Live activity events |
| `INCIDENT_AGENT_LIVE_APPROVALS` | `api/db_ops/live_status.py` | Pending/resolved approvals |

---

## Manual Testing

### Health check

```bash
curl http://localhost:8080/api/v1/health
```

### Process incident manually

```bash
curl -X POST http://localhost:8080/api/v1/incidents/process \
  -H "Content-Type: application/json" \
  -d '{
    "incident_object": {
      "requestor": "Test User (testuser)",
      "description": "Test incident for manual processing",
      "number": "INC99999999",
      "state": "Awaiting Assignment",
      "service_offering": "Data Integration Platform Services",
      "assignment_group": "EDS-PLATFORM-SNOWFLAKE",
      "assigned_to": "",
      "reopen_count": "0"
    },
    "source": "manual"
  }'
```

### Legacy script entry point

```bash
python call_incident_agent.py
```

### ESP client (manual)

```bash
python -m agent.esp.update
python -m agent.esp.resolve
```

### View activities

```bash
curl http://localhost:8080/api/v1/activities?limit=20
```

### Consumer status

```bash
curl http://localhost:8080/api/v1/consumer/status
```

---

## Troubleshooting

| Issue | Possible cause | Fix |
|---|---|---|
| Consumer not running | CyberArk cert/password failure | Verify `[KAFKA_CONSUMER]` cert paths in `config.ini` and network to CCP |
| `API call failed` in logs | Wrong `base_url` / port mismatch | Set `[APP]` `base_url` to match running port |
| AI agent errors | Wrong URL or token | Check `[AGENT_API_CONFIG]` in `config.ini` |
| ServiceNow update fails | OAuth credentials expired | Refresh `[ESP_INCIDENT]` client_id/secret |
| No Kafka messages | Wrong topic/group or filter | Verify `[KAFKA_CONSUMER]` `topic_to_consume` / `consumer_group_Id` and `SERVICE_OFFERINGS` |
| Incident stored but no workflow | Assignment group not in `FILTER_ASSIGNMENT_GROUPS`, no `DUTY_ROSTERS` entry, or service not restarted after the change | Add the group to `FILTER_ASSIGNMENT_GROUPS` and a `DUTY_ROSTERS` key in `agent/assignment_scope.py` and to `agent/platform.py`, restart the service, and check the startup `config_warning` events |
| UI not updating | SSE disconnected | Refresh browser; check `/api/v1/activities/stream` |
| MongoDB errors | Connection string or TLS | Verify `[MONGO_DETAILS]` and network access |

Check `logs/incident_agent.log` and the live dashboard for detailed error events.

---

## Legacy Compatibility

| Legacy file | Current behavior |
|---|---|
| `call_incident_agent.py` | Wraps `process_incident()` from `agent/graph.py` |
| `consumer.py` (old standalone) | Now imported by `main.py`; Kafka settings come from `config.ini` `[KAFKA_CONSUMER]` |
| `util.py` | Deprecated; Kafka/CyberArk settings moved to `config.ini` `[KAFKA_CONSUMER]` |
| `update_incident.py` | Moved to `agent/esp/update.py`; used by `update_incident_tool` |
| `resolve_incident.py` | Moved to `agent/esp/resolve.py`; used by `resolve_incident_tool` |
| `db_ops.py` | Moved to `api/db_ops/` (`client.py`, `incidents.py`, `live_status.py`) |
| `api/routes.py` | Removed; handlers live in `api/controller/` and are mounted from `main.py` |
| `services/` | Moved to `api/services/` |

To run only the API without Kafka (for local dev), you can temporarily comment out `start_kafka_consumer()` in `main.py` lifespan, or process incidents via the REST API.

---

## Security Notes

- Store credentials in environment variables or a secrets manager; avoid committing `config.ini` with real secrets.
- CyberArk certificates must be kept secure and rotated per policy.
- The AI agent API token and ServiceNow OAuth secrets should be rotated regularly.
- Run the service behind HTTPS in production.
- Restrict network access to Kafka, MongoDB, and ServiceNow endpoints.

---

## License

*(The License section heading is visible at the bottom of the last photographed page; its content was not captured.)*
