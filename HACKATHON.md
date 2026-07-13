# FlowMap Agent — Slack Agent Builder Challenge

> **Track:** New Slack Agent

## What is FlowMap Agent?

FlowMap Agent is a Slack bot that **diagnoses production issues and auto-generates fixes** by combining cross-repo code intelligence with observability data.

Ask it a question in Slack:

> @FlowMap Agent why is txn 334664236896620544 not found?

It will:
1. **Search logs** across all microservices via Grafana/Loki
2. **Read source code** across repos via FlowMap (tree-sitter AST + vector search)
3. **Trace the root cause** — e.g., IEEE 754 precision loss in `order.ts:65`
4. **Generate a fix** with TDD and push a PR to GitHub

---

## Technologies Used

| Technology | How We Use It |
|------------|---------------|
| **MCP Server Integration** | FlowMap MCP (code intelligence), Grafana MCP (Loki/Prometheus queries), Slack MCP (thread history) |
| **Slack AI Capabilities** | LLM-driven agent loop with native tool calling via Ollama |
| **Slack Agent Builder** | Slack Bolt + Socket Mode for real-time event handling |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Slack Workspace                         │
│                                                             │
│   User: @FlowMap Agent why is txn 334... not found?         │
│                                                             │
│   Bot: 🔍 Root cause: IEEE 754 at order.ts:65              │
│        [Create Bugfix PR]  [Show History]  [Explain Code]   │
└──────────────────────────┬──────────────────────────────────┘
                           │ Socket Mode (WebSocket)
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                    FlowMap Agent                              │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ SlackBot │→ │  Agent   │→ │  LLM     │ (Ollama Cloud)    │
│  │ (bolt)   │  │ (loop)   │  │ (gemma4) │                   │
│  └──────────┘  └────┬─────┘  └──────────┘                   │
│                     │                                        │
│         ┌───────────┼───────────┬──────────────┐             │
│         ▼           ▼           ▼              ▼             │
│  ┌────────────┐ ┌─────────┐ ┌──────────┐ ┌──────────┐      │
│  │ FlowMap    │ │ Grafana │ │ Slack    │ │ GitOps   │      │
│  │ MCP        │ │ MCP     │ │ MCP      │ │ (SSH)    │      │
│  │            │ │         │ │          │ │          │      │
│  │ • search   │ │ • Loki  │ │ • history│ │ • branch │      │
│  │ • cat      │ │ • Prom  │ │ • search │ │ • commit │      │
│  │ • history  │ │ • list  │ │          │ │ • push   │      │
│  └─────┬──────┘ └────┬────┘ └─────┬────┘ └──────────┘      │
│        │             │            │                          │
└────────┼─────────────┼────────────┼──────────────────────────┘
         ▼             ▼            ▼
┌──────────────┐ ┌──────────┐ ┌──────────────────┐
│  FlowMap     │ │ Grafana  │ │ Slack MCP Server │
│  (LanceDB +  │ │ (Docker) │ │ (port 13080)     │
│   tree-sitter│ │          │ │                  │
│   + ripgrep) │ │ Loki     │ │ slack-mcp-server │
└──────────────┘ │ Prom     │ └──────────────────┘
                 │ Tempo    │
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

See `diagram.html` for the full sequence diagram.

---

## How It Works

### 1. Code Intelligence (FlowMap MCP)

FlowMap indexes codebases using **tree-sitter AST parsing** and stores embeddings in **LanceDB**. It provides:

- **4-way hybrid search**: ripgrep keyword + BM25 + vector similarity + symbol lookup
- **Reciprocal Rank Fusion** merges results with query-type-aware weights
- **Cross-repo search** across Python, TypeScript, Go, Java, Swift, and more

Exposed as MCP tools: `flowmap_search`, `flowmap_cat`, `flowmap_history`, `flowmap_repos`

### 2. Observability (Grafana MCP)

The agent queries Grafana's observability stack via **mcp-grafana**:

- **Loki**: Log queries across all microservices
- **Prometheus**: Metrics and counters
- **Datasource discovery**: Auto-detect available data sources

Exposed as MCP tools: `grafana_query_loki`, `grafana_query_prometheus`, `grafana_list_datasources`

### 3. Thread Context (Slack MCP)

When mentioned in a thread, the agent fetches conversation history via **Slack MCP**:

- Provides context for diagnosis (e.g., "payment failed for txn 334...")
- Passes thread context to LLM for fix generation
- Posts PR link back to the thread

Exposed as MCP tools: `slack_conversations_history`, `slack_conversations_replies`

### 4. LLM Agent Loop

The agent uses **Ollama Cloud** (`gemma4:31b-cloud`) with native tool calling:

- LLM decides which tools to call and when to stop
- Minimum 5 tool calls before accepting a diagnosis
- System prompt enforces root cause analysis discipline
- No hardcoded orchestration logic

### 5. TDD Bugfix PR Generation

When the user clicks "Create Bugfix PR":

1. **RED Phase**: Generate failing test (with export map + thread context) → run test → FAIL
2. **GREEN Phase**: Generate fix → validate syntax → run test → PASS
3. **Push**: Branch → commit → push → PR URL

---

## Demo Scenario

### The Bug

A polyglot payment microservice system:
- **Payment Service** (Python): generates Snowflake IDs like `334664236896620544`
- **Order Service** (TypeScript): receives orders, forwards to ledger
- **Ledger Service** (Go): records and looks up transactions

**The bug**: JavaScript's `JSON.parse` silently rounds 17-18 digit IDs due to IEEE 754 precision loss.

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

The PR includes both the test and the fix.

---

## Setup

### Prerequisites

- Python 3.11+
- Docker + Docker Compose
- Node.js 20+ (for demo services)
- Ollama (for local embeddings)

### Quick Start

```bash
# 1. Clone and install
git clone https://github.com/aniketcodes/repo-embedd.git
cd repo-embedd
pip install -e .

# 2. Start Ollama (for embeddings only — LLM is cloud)
ollama serve
ollama pull qwen3-embedding:0.6b

# 3. Start demo infrastructure
cd demo-repos && docker compose up -d

# 4. Index repos
flowmap index

# 5. Configure environment
cp .env.example .env
# Edit .env with your Slack tokens

# 6. Start bot
PYTHONPATH=packages/agent/src:packages/core/src:packages/slack/src \
  OLLAMA_MODEL=gemma4:31b-cloud \
  python run_bot.py
```

### Environment Variables

```bash
# Slack
SLACK_SIGNING_SECRET=...
SLACK_BOT_TOKEN=xoxb-...
SLACK_APP_TOKEN=xapp-...

# LLM (cloud — no local GPU needed)
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:31b-cloud

# Grafana
GRAFANA_URL=http://localhost:3000
GRAFANA_SERVICE_ACCOUNT_TOKEN=...
ENABLE_GRAFANA_ADAPTER=1
```

---

## MCP Integration Details

### FlowMap MCP Server

- **Transport**: SSE on port 3001
- **Library**: `mcp` Python package (`FastMCP`)
- **Tools**: `flowmap_search`, `flowmap_cat`, `flowmap_history`, `flowmap_repos`, `flowmap_symbols`, `flowmap_map`

### Grafana MCP Server

- **Transport**: stdio (JSON-RPC 2.0)
- **Server**: `grafana/mcp-grafana` Docker container
- **Tools**: `grafana_query_loki`, `grafana_query_prometheus`, `grafana_list_datasources`, `grafana_search_dashboards`

### Slack MCP Server

- **Transport**: SSE on port 13080
- **Server**: `slack-mcp-server` npm package
- **Tools**: `slack_conversations_history`, `slack_conversations_replies`, `slack_users_info`

All integrations are unified through the `MCPAdapter` abstract base class with O(1) tool routing.

---

## Project Structure

```
repo-embedd/
├── run_bot.py                    # Bot entry point (Socket Mode)
├── flowmap/                      # Code intelligence engine
│   ├── cli.py                    # CLI (11 commands)
│   ├── search/hybrid.py          # 4-way RRF search
│   └── parsing/chunker.py        # Tree-sitter AST chunking
├── packages/agent/src/agent/
│   ├── agent.py                  # LLM agent loop
│   ├── bot.py                    # Slack handler
│   ├── actions.py                # PR generation (TDD + quick)
│   ├── grafana_adapter.py        # Grafana MCP adapter
│   ├── flowmap_adapter.py        # FlowMap MCP adapter
│   └── slack_adapter.py          # Slack MCP adapter
└── demo-repos/                   # Demo microservices
    ├── docker-compose.yml
    ├── demo-payment-service/     # Python
    ├── demo-order-service/       # TypeScript (bug here)
    ├── demo-ledger-service/      # Go
    └── demo-observability/       # Grafana stack
```

---

## Judging Criteria

### Technological Implementation
- Uses **3 MCP servers** (FlowMap, Grafana, Slack)
- **LLM-driven agent loop** with native tool calling
- **TDD-based bugfix generation** with syntax validation
- **4-way hybrid search** with Reciprocal Rank Fusion

### Design
- Clean Slack UX with streaming updates and action buttons
- Agent decides tool calls — no hardcoded orchestration
- Export map ensures correct import paths in generated code
- Retry loop with error feedback for fix generation

### Potential Impact
- Reduces debugging time from hours to seconds
- Works across polyglot microservices (Python, TypeScript, Go)
- Auto-generates tested fixes, not just diagnosis
- Deployable to any Slack workspace

### Quality of the Idea
- Combines code intelligence + observability + LLM in one agent
- Cross-repo search finds bugs that span multiple services
- TDD approach ensures fixes are verified before PR
- Request ID tracing detects data corruption across services

---

## Links

- **GitHub**: https://github.com/aniketcodes/repo-embedd
- **Architecture Diagram**: `diagram.html`
- **Sequence Diagram**: `SUBMISSION.md`

---

## Team

Built for the Slack Agent Builder Challenge 2026.
