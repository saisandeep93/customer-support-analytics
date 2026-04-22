# customer-support-analytics
# Intelligent Customer Support Analytics Platform

An end-to-end AI-powered customer support system built on Claude API, ChromaDB, and Streamlit. Demonstrates RAG pipelines, agentic AI with multi-tool reasoning, MCP server implementation, and full observability — applied to a QSR food delivery brand.

---

## Architecture Overview

```
Customer Message
       │
       ▼
┌─────────────────────────────────────────────────┐
│              Master Router (main.py)             │
│         Haiku classifier + signal detection      │
└──────────┬──────────────┬───────────────────────┘
           │              │                │
           ▼              ▼                ▼
    ┌────────────┐  ┌───────────┐  ┌────────────────┐
    │ TIER 1     │  │ TIER 2    │  │ TIER 3         │
    │ FAQ Lookup │  │ RAG       │  │ ReAct Agent    │
    │ (free)     │  │ Pipeline  │  │ (Sonnet)       │
    │            │  │ (Haiku)   │  │                │
    └─────┬──────┘  └─────┬─────┘  └──────┬─────────┘
          │               │               │
          │               │         ┌─────▼──────────┐
          │               │         │  5 Tools        │
          │               │         │  • customer     │
          │               │         │    profile      │
          │               │         │  • order detail │
          │               │         │  • complaint    │
          │               │         │    history      │
          │               │         │  • KB search    │
          │               │         │  • escalate     │
          │               │         └─────────────────┘
          │               │               │
          └───────────────┴───────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │   interaction_logs    │
              │   (SQLite)            │
              └───────────┬───────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Streamlit Dashboard  │
              │  6 evaluation panels  │
              └───────────────────────┘

MCP Server runs in parallel — exposes all tools
through standard protocol for any MCP client
```

---

## Seven Layers

### Layer 1 — Data Foundation
SQLite database with 8 tables covering customers, orders, products, support tickets, knowledge base documents, and interaction logs. Schema designed to support the full analytics pipeline from day one.

### Layer 2 — RAG Pipeline
Knowledge base documents chunked into 500-character pieces with 100-character overlap, embedded using `all-MiniLM-L6-v2` from Hugging Face, and stored in ChromaDB. At query time, the customer's message is embedded and the top-N most semantically similar chunks are retrieved and passed to Claude Haiku alongside the customer profile.

### Layer 3 — Deterministic Workflow
A Haiku-powered query classifier routes simple, factual queries to a separate FAQ vector store. FAQ answers are returned directly without an LLM generation step — zero cost beyond the classifier call. A similarity threshold of 0.55 prevents wrong-match responses.

### Layer 4 — Agentic Workflow
Complex queries enter a ReAct loop (Reason → Act → Observe) powered by Claude Sonnet. The agent has access to five tools and decides at runtime which to call, in what order, and when it has enough information to respond. A max_iterations=10 guard prevents runaway loops.

### Layer 5 — MCP Server + Master Router
All five agent tools are exposed through an MCP server using Anthropic's Model Context Protocol SDK. Any MCP-compatible client — including Claude Desktop — can discover and call these tools through the standard interface without custom integration code.

The master router (`main.py`) is the single entry point that connects all three tiers, routing every query to the appropriate execution path.

### Layer 6 — Observability and Evaluation
Every interaction writes a structured log row capturing: query text, routing tier, retrieved document IDs, similarity scores, tools called, token consumption, latency, resolution status, escalation flag, and customer feedback score. Six evaluation metrics are computed from these logs.

### Layer 7 — Analytics Dashboard
A Streamlit dashboard with six panels: system overview (north star metrics), routing distribution, cost analysis by tier, latency distribution (P50/P95/P99), customer feedback, and a searchable interaction log with drill-down detail view.

---

## Tech Stack

| Component | Technology |
|---|---|
| LLM — standard queries | Claude Haiku (`claude-haiku-4-5-20251001`) |
| LLM — complex reasoning | Claude Sonnet (`claude-sonnet-4-6`) |
| Vector store | ChromaDB (persistent, cosine similarity) |
| Embedding model | `all-MiniLM-L6-v2` (Hugging Face, local) |
| MCP server | Anthropic MCP SDK |
| Database | SQLite |
| Dashboard | Streamlit + Plotly |
| Language | Python 3.11 |

---
## Two Implementation Approaches

Each core component is implemented twice — once from primitives 
to demonstrate understanding of the mechanics, once using 
industry-standard frameworks for production readiness.

| Component | Primitive | Framework |
|---|---|---|
| RAG Pipeline | `src/rag_pipeline.py` | `src/langchain_rag.py` |
| Agent | `src/agent.py` | `src/langgraph_agent.py` |

The primitive implementations use direct Anthropic SDK calls, 
manual ChromaDB integration, and a hand-written ReAct loop. 
The framework implementations use LangChain LCEL chains, 
LangGraph's create_react_agent, and the @tool decorator pattern.


## Key Design Decisions

**Three-tier routing over single-path RAG**
Running every query through the full RAG pipeline wastes tokens and adds latency for queries that need no personalisation. The three-tier router reduces average cost per interaction by ~80% compared to agent-only routing:

```
FAQ lookup:    ~185 tokens   $0.00015 per interaction
RAG pipeline:  ~800 tokens   $0.00070 per interaction
Agent:        ~9,000 tokens  $0.028  per interaction
```

**RAG as an agent tool, not an alternative**
RAG and agentic AI solve different problems. RAG grounds responses in unstructured knowledge (policies, FAQs). The agent handles structured data lookups (orders, profiles) and multi-step reasoning. In this system, `search_knowledge_base` is one of the agent's five tools — RAG operates inside the agent's reasoning loop.

**Chunking strategy**
500-character chunks with 100-character overlap. Chunk size chosen to capture complete policy rules without mixing unrelated content. Overlap preserves sentence boundary context across chunk seams. Document title prepended to every chunk to preserve topic identity in short chunks.

**Model selection by task complexity**
Haiku for classification and standard generation — fast, cheap, accurate for structured tasks. Sonnet for agentic reasoning — required for reliable multi-step tool selection and complex judgment calls. No Opus — the task complexity does not justify the cost premium.

**Observability from day one**
Every interaction is logged before the response is returned. The interaction_logs table is designed as an event store — one row per interaction, all metadata captured — so evaluation metrics can be computed at any point without re-running interactions.

---

## Evaluation Metrics

Six metrics computed from interaction_logs:

| Metric | What It Measures |
|---|---|
| Resolution rate | % of interactions resolved without human escalation |
| Routing distribution | % of traffic hitting each tier — cost optimisation signal |
| Retrieval quality | Avg similarity score of retrieved chunks — KB health signal |
| Cost analysis | Token consumption and estimated USD cost by tier |
| Latency distribution | P50 / P95 / P99 response times — SLA monitoring |
| Feedback summary | Customer satisfaction scores where collected |

---

## Project Structure

```
customer-support-analytics/
├── data/
│   ├── support.db          ← SQLite database
│   └── chroma_db/          ← ChromaDB vector store
├── src/
│   ├── create_schema.py    ← Layer 1: database schema
│   ├── seed_data.py        ← Layer 1: realistic seed data
│   ├── queries.py          ← Layer 1: SQL query functions
│   ├── rag_pipeline.py     ← Layer 2: RAG implementation
│   ├── deterministic_workflow.py ← Layer 3: FAQ routing
│   ├── agent.py            ← Layer 4: ReAct agent
│   ├── mcp_server.py       ← Layer 5: MCP server
│   ├── main.py             ← Layer 5: master router
│   ├── observability.py    ← Layer 6: logging and metrics
│   └── dashboard.py        ← Layer 7: Streamlit dashboard
├── docs/
├── .env                    ← API keys (not committed)
├── .gitignore
└── README.md
```

---

## Running the Project

**Setup**
```bash
git clone https://github.com/yourusername/customer-support-analytics.git
cd customer-support-analytics
python -m venv venv
source venv/bin/activate
pip install anthropic chromadb sentence-transformers python-dotenv numpy mcp streamlit plotly
```

**Configure API key**
```bash
echo "ANTHROPIC_API_KEY=your-key-here" > .env
```

**Initialise database**
```bash
python src/create_schema.py
python src/seed_data.py
```

**Test each layer**
```bash
python src/rag_pipeline.py          # Layer 2: RAG pipeline
python src/deterministic_workflow.py # Layer 3: FAQ routing
python src/agent.py                  # Layer 4: agent
python src/main.py                   # Layer 5: master router
python src/observability.py          # Layer 6: generate logs and metrics
```

**Launch dashboard**
```bash
streamlit run src/dashboard.py
```

**Start MCP server**
```bash
python src/mcp_server.py
```

---

## What This Project Demonstrates

Built to bridge the gap between CDP/analytics leadership experience and modern AI engineering requirements. Every component is built from primitives — no black-box frameworks — so the mechanics of each layer can be explained from first principles in an interview.

Concepts covered: RAG pipelines, vector stores and embeddings, chunking strategy and trade-offs, prompt engineering and augmentation, deterministic vs non-deterministic workflows, agentic AI and the ReAct loop, tool use and function calling, MCP protocol and tool standardisation, LLM observability and structured logging, AI system evaluation metrics, human-in-the-loop feedback, cost and latency trade-off analysis, and analytics dashboard design for AI systems.

---

