# Project Story

## The Inspiration

Every engineer knows this moment.

A production alert goes off. Customers are waiting. The pressure is on.

You open Grafana. Then Loki. Then GitHub. Then your IDE. Five minutes later you have fifteen browser tabs open, three terminal windows, and a growing collection of copied request IDs. You're constantly switching contexts, trying to stitch together a story that no single tool can tell.

The code lives in one place.
The logs live somewhere else.
The discussion is buried in a Slack thread.

The hardest part isn't finding information—it's connecting it.

The turning point came while debugging a real production issue in a polyglot payment system built with Python, TypeScript, and Go.

A Snowflake ID (`334664236896620544`) entered one service correctly but emerged from another as `334664236896620540`. JavaScript had silently rounded the value because it exceeded `Number.MAX_SAFE_INTEGER`. Every service looked correct in isolation. The logs disagreed. The source code was spread across multiple repositories. Finding the exact line where the corruption occurred took nearly **45 minutes**.

The fix itself took less than two.

That experience led to a simple question:

**If an engineer can eventually connect all these pieces, why can't an AI agent do it in seconds?**

---

## How We Built It

FlowMap Agent gives AI the same capabilities an experienced engineer uses during an incident—but without the context switching.

### 1. It understands the code

FlowMap indexes repositories using Tree-sitter, Ollama embeddings, and LanceDB.

Instead of relying on a single search technique, it combines keyword search, BM25, vector similarity, and symbol lookup using Reciprocal Rank Fusion. The result is accurate code retrieval across multiple repositories, exposed through an MCP server.

### 2. It understands what's happening in production

Through Grafana MCP, the agent retrieves logs from Loki and metrics from Prometheus.

Rather than chasing isolated error messages, it follows a **request_id** across services to reconstruct the entire execution path and identify exactly where data changes.

### 3. It understands the conversation

Using Slack MCP, the agent reads the thread where it was mentioned.

This gives it the missing business context—what failed, who reported it, and what the team has already investigated—before it starts reasoning.

---

## The Agent Loop

Instead of following a scripted workflow, the LLM decides which tools to use and when.

We discovered that powerful agents need strong guardrails.

Our system prompt teaches the agent how experienced engineers debug distributed systems:

* An error message is a symptom—not the root cause.
* Search across every repository, not just the service that failed.
* Trace the producer before blaming the consumer.
* Follow the request ID before following assumptions.
* Stop only when you've explained **why**, not just **what**.

The result is an agent that investigates rather than simply answers.

---

## From Diagnosis to Resolution

Finding the bug is only half the job.

With one click, FlowMap Agent can create a bug-fix pull request by:

* Reading the relevant source code
* Building an export map for correct imports
* Generating a failing test (RED)
* Implementing the fix (GREEN)
* Validating syntax
* Pushing the change to GitHub

The workflow follows TDD because confidence matters as much as speed.

---

## What We Learned

Building autonomous debugging agents taught us lessons we didn't expect.

**MCP is incredibly powerful—but orchestration matters.**

We integrated three MCP servers using different transports (SSE, stdio JSON-RPC, and in-process). An adapter layer made them feel like one system while hiding implementation differences from the agent.

**LLMs don't just need tools—they need engineering discipline.**

Without explicit reasoning rules, the agent would stop after the first matching log line. Once we taught it to search across repositories and verify assumptions, its investigations became dramatically more reliable.

**Context beats intelligence.**

The biggest improvement didn't come from a larger model. It came from giving the model the right context—from code, production telemetry, and Slack—at exactly the right time.

**Real bugs are rarely isolated.**

The IEEE 754 rounding issue wasn't invented for a demo. It was a real production problem where every individual component behaved "correctly," but the system as a whole failed. Those are exactly the kinds of incidents FlowMap Agent is built to solve.

---

## Why This Matters

Modern software isn't difficult because the code is complex.

It's difficult because the knowledge needed to solve a problem is scattered across repositories, dashboards, logs, documentation, and conversations.

FlowMap Agent brings those worlds together.

Instead of asking engineers to hunt for answers, it gives AI enough context to investigate like an experienced teammate—turning hours of context switching into seconds of focused reasoning.

Built for the **Slack Agent Builder Challenge 2026**.
