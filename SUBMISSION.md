# FlowMap Agent — Slack Hackathon Submission

## What is it?

FlowMap Agent is a Slack bot that **diagnoses production issues and auto-generates fixes** by combining cross-repo code intelligence with observability data (Grafana/Loki/Prometheus).

Ask it a question like:
> @FlowMap Agent why is txn 334664236896620544 not found?

It will:
1. **Search logs** across all microservices via Grafana/Loki
2. **Read source code** across repos via FlowMap (tree-sitter AST + vector search)
3. **Trace the root cause** — e.g., IEEE 754 precision loss in `order.ts:65`
4. **Generate a fix** and push a PR to GitHub

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Slack Workspace                          │
│                                                                 │
│   User: @FlowMap Agent why is txn 334664236896620544 not found? │
│                                                                 │
│   Bot: 🔍 Diagnosis: IEEE 754 precision loss in order.ts:65     │
│        [Create PR]  [Show History]  [Explain Code]              │
└──────────────────────────┬──────────────────────────────────────┘
                           │ Socket Mode (WebSocket)
                           ▼
┌──────────────────────────────────────────────────────────────────┐
│                      FlowMap Agent (run_bot.py)                  │
│                                                                  │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ SlackBot │  │  Agent   │  │  LLM     │  │ ActionPlanner    │ │
│  │ (bolt)   │→ │ (loop)   │→ │ (Ollama) │  │ (recommend PR)   │ │
│  └──────────┘  └────┬─────┘  └──────────┘  └──────────────────┘ │
│                     │                                            │
│         ┌───────────┼───────────┬──────────────┐                 │
│         ▼           ▼           ▼              ▼                 │
│  ┌────────────┐ ┌─────────┐ ┌──────────┐ ┌──────────────┐      │
│  │ FlowMap    │ │ Grafana │ │ Slack    │ │ GitOps       │      │
│  │ Adapter    │ │ Adapter │ │ Adapter  │ │ (SSH push)   │      │
│  │            │ │         │ │          │ │              │      │
│  │ • search   │ │ • Loki  │ │ • history│ │ • branch     │      │
│  │ • cat      │ │ • Prom  │ │ • search │ │ • commit     │      │
│  │ • history  │ │ • list  │ │ • channels│ │ • push      │      │
│  │ • repos    │ │ • dash  │ │          │ │ • PR URL     │      │
│  └─────┬──────┘ └────┬────┘ └─────┬────┘ └──────────────┘      │
│        │             │            │                              │
└────────┼─────────────┼────────────┼──────────────────────────────┘
         │             │            │
         ▼             ▼            ▼
┌──────────────┐ ┌──────────┐ ┌──────────────────┐
│  FlowMap     │ │ Grafana  │ │ Slack MCP Server │
│  (LanceDB +  │ │ (Docker) │ │ (port 13080)     │
│   Ollama +   │ │          │ │                  │
│   tree-sitter│ │ Loki     │ │ slack-mcp-server │
│   + ripgrep) │ │ Prom     │ │ npm package      │
└──────────────┘ │ Tempo    │ └──────────────────┘
                 └──────────┘
                        │
                        ▼
                 ┌──────────────┐
                 │ Demo Services│
                 │ (Docker)     │
                 │              │
                 │ • payment (Py)│
                 │ • order (TS) │
                 │ • ledger (Go)│
                 └──────────────┘
```

---

## How it works

### 1. Code Intelligence (FlowMap)
- **Tree-sitter AST parsing** extracts functions, classes, methods with signatures
- **Ollama embeddings** (`qwen3-embedding:0.6b`) stored in **LanceDB**
- **4-way hybrid search**: ripgrep keyword + BM25 + vector similarity + symbol lookup
- **Reciprocal Rank Fusion** merges results with query-type-aware weights
- Supports: Python, TypeScript, JavaScript, Go, Java, Swift, YAML, JSON

### 2. Observability (Grafana MCP)
- **mcp-grafana** Docker container exposes Loki/Prometheus queries via MCP
- Agent queries logs across all services, correlates by `request_id`
- Detects ID modifications across service boundaries

### 3. LLM Agent (Ollama)
- **gemma4:31b-cloud** runs locally via Ollama
- Native tool calling — LLM decides which tools to call and when to stop
- Minimum 5 tool calls before accepting a diagnosis
- System prompt enforces root cause analysis discipline

### 4. PR Generation
- LLM reads the source code and generates a fix
- Syntax validation (TypeScript/Python/Go)
- Git branch → commit → push via SSH
- Returns GitHub PR creation URL

---

## Demo Walkthrough

### The Bug
The demo system is a polyglot payment microservice:
- **Payment Service** (Python): generates Snowflake IDs like `334664236896620544`
- **Order Service** (TypeScript): receives orders, forwards to ledger
- **Ledger Service** (Go): records and looks up transactions

The bug: JavaScript's `JSON.parse` silently rounds 17-18 digit IDs due to IEEE 754 precision loss (`Number.MAX_SAFE_INTEGER` = 2^53-1 ≈ 9×10^15).

### The Diagnosis
```
User: @FlowMap Agent why is txn 334664236896620544 not found?

Bot:
• Root Cause: demo-order-service rounds the transaction ID
• Evidence:
  - Payment log: txn_id: 334664236896620544 (original)
  - Order log:   txn_id: 334664236896620540 (rounded)
  - Ledger log:  txn not found
• Location: demo-order-service/src/order.ts:65
```

### The Fix
```diff
- const txnId = req.body.transaction_id;
+ const txnId = String(req.body.transaction_id || '');
```

---

## MCP Integration

FlowMap Agent uses the **Model Context Protocol (MCP)** in three ways:

| Integration | Transport | Purpose |
|-------------|-----------|---------|
| FlowMap MCP Server | SSE (port 3001) | Expose code intelligence tools |
| Slack MCP Client | SSE (port 13080) | Access Slack data via MCP |
| Grafana MCP | stdio (JSON-RPC) | Query Loki/Prometheus |

All integrations are unified through the `MCPAdapter` abstract base class with O(1) tool routing.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Ollama Cloud (`gemma4:31b-cloud`) — no local GPU needed |
| Code Intelligence | tree-sitter + LanceDB + Ollama embeddings (`qwen3-embedding:0.6b`, local) |
| Observability | Grafana + Loki + Prometheus + Tempo |
| Slack Integration | slack-bolt + Socket Mode + MCP |
| Microservices | Python + TypeScript + Go |
| Containerization | Docker Compose |
| Version Control | Git + SSH |

---

## Environment Variables

```bash
# Slack
SLACK_SIGNING_SECRET=...
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...

# LLM
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:31b-cloud

# Grafana
GRAFANA_URL=http://localhost:3000
GRAFANA_SERVICE_ACCOUNT_TOKEN=...
ENABLE_GRAFANA_ADAPTER=1
```

---

## Running Locally

```bash
# 1. Start Ollama (for embeddings only — LLM is cloud-hosted)
ollama serve
ollama pull qwen3-embedding:0.6b

# 2. Start demo infrastructure
cd demo-repos && docker compose up -d

# 3. Index repos
flowmap index

# 4. Start bot
PYTHONPATH=packages/agent/src:packages/core/src:packages/slack/src \
  OLLAMA_MODEL=gemma4:31b-cloud \
  python run_bot.py
```

### Resource Requirements

| Resource | Requirement |
|----------|-------------|
| RAM | ~2 GB (embeddings only — LLM is cloud) |
| CPU | Any modern CPU |
| Disk | ~500 MB (LanceDB index + Docker images) |
| Network | Internet access (for Ollama cloud LLM + Slack) |

---

## Files

```
flowmap-cli/
├── run_bot.py                    # Bot entry point
├── flowmap/                      # Code intelligence engine
│   ├── cli.py                    # CLI (11 commands)
│   ├── search/hybrid.py          # 4-way RRF search
│   └── parsing/chunker.py        # Tree-sitter AST chunking
├── packages/agent/src/agent/
│   ├── agent.py                  # LLM agent loop
│   ├── bot.py                    # Slack handler
│   ├── actions.py                # PR generation
│   ├── grafana_adapter.py        # Grafana MCP
│   └── flowmap_adapter.py        # FlowMap MCP
└── demo-repos/                   # Demo microservices
    ├── docker-compose.yml
    ├── demo-payment-service/     # Python
    ├── demo-order-service/       # TypeScript (bug here)
    ├── demo-ledger-service/      # Go
    └── demo-observability/       # Grafana stack
```
