# 💼 AI Portfolio Advisory Copilot

A multi-agent system that analyses and manages an **Alpaca paper-trading portfolio** through the **Alpaca MCP server**.
It combines **SEC 10-K research** with **technical, risk and portfolio-management agents**. Every proposal is checked
against the investor's mandate by a **deterministic compliance engine**, and an order is sent only after **human approval**.
Everything runs on a **local LLM** through Ollama, and the same system is implemented in **Pydantic AI** and **LangChain**
so the two frameworks can be compared on the same evaluation set.

> Educational project. Paper trading only — not investment advice.

## Architecture

```
          Alpaca paper account (MCP)                         data/10k PDFs
   account · positions · clock · daily bars · news       (Apple, Alphabet, Tesla)
                 │                                                 │
                 ▼                                                 ▼
  ┌──────────────────────────────── for each stock ────────────────────────────────┐
  │  🔎 Research analyst   10-K search (RAG) + Alpaca news (MCP)  → FundamentalView  │
  │  📈 Technical analyst  Python indicators + Alpaca snapshot    → TechnicalView    │
  │  ⚠️ Risk manager       views + portfolio + investor profile   → RiskView         │
  │  👔 Portfolio manager  everything above                       → TradeProposal    │
  └─────────────────────────────────────────────────────────────────────────────────┘
                 │
                 ▼
  🛡️ Compliance engine (Python): suitability · concentration · cash buffer · order size · no shorting
                 │
                 ▼
  👤 Human approval ──► place_stock_order (MCP, paper) ──► 📒 JSONL audit trail
```

## Design decisions

| Decision | Why |
|---|---|
| **LLMs propose, code disposes** | Agents only *propose* trades. `check_trade()` blocks unsuitable trades and resizes oversized ones, so a model mistake cannot become an unsafe order. |
| **Least privilege** | Agents get two read-only MCP tools (`get_stock_snapshot`, `get_news`). Orders are placed by Python, never by an agent. |
| **Paper trading forced in code** | `alpaca_env()` always sets `ALPACA_PAPER_TRADE=true`, whatever `.env` says. |
| **Human in the loop** | Dry run by default. With `EXECUTE_ORDERS = True`, every order still needs a typed `yes`. |
| **Numbers computed in Python** | RSI, SMA, MACD and volatility are calculated in code. LLMs receive the numbers instead of doing the arithmetic. |
| **Prompt-injection awareness** | News is treated as untrusted data. The Alpaca server also wraps tool output in a trust-boundary envelope. |
| **Local and private** | LLM (`nemotron-3.5-lightning:30b-mlx`) and embeddings (`nomic-embed-text`) run on Ollama. No data leaves the machine except the Alpaca API calls. |
| **Fair framework comparison** | Data, prompts, schemas, rules and evaluation live in `portfolio_core.py`. The notebooks contain only framework-specific code. |

## Evaluation

There are eight fixed scenarios with expected outcomes. For example: a conservative investor must not buy a
65%-volatility stock, a held stock turning bearish should be sold, and short selling is never allowed. They are scored on:

* **system accuracy**: the final decision after the rules is the expected one
* **LLM accuracy**: the model's raw proposal was already correct *and* compliant
* **rule interventions**: how often the engine had to block or resize a proposal
* **errors** (schema or parsing failures) and **latency**

Results are written to `outputs/eval_<framework>.json`. `compare_frameworks()` shows them side by side.

| Framework | System accuracy | LLM accuracy | Rule interventions | Errors | Avg s / scenario |
|---|---|---|---|---|---|
| Pydantic AI | _run the notebook_ | | | | |
| LangChain | _run the notebook_ | | | | |

## Project structure

```
portfolio/
├── portfolio_core.py           # shared: Alpaca helpers, indicators, 10-K RAG, profiles, schemas, prompts, rules, evaluation
├── copilot_pydantic_ai.ipynb   # Pydantic AI version (OllamaProvider, MCPServerStdio)
├── copilot_langchain.ipynb     # LangChain version (ChatOllama, langchain-mcp-adapters, create_agent)
├── data/10k/                   # 2024 Form 10-K PDFs (AAPL, GOOGL, TSLA)
├── outputs/                    # audit trails and evaluation results (created at runtime)
├── requirements.txt
└── .env.example
```

## Setup

1. **Alpaca paper account**: create paper-trading API keys at [alpaca.markets](https://alpaca.markets), then
   `cp .env.example .env` and fill in `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`.
2. **uv**, which runs the official Alpaca MCP server with `uvx alpaca-mcp-server`:
   `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. **Ollama**:
   ```bash
   ollama pull nemotron-3.5-lightning:30b-mlx
   ollama pull nomic-embed-text
   OLLAMA_CONTEXT_LENGTH=16384 ollama serve
   ```
4. `pip install -r requirements.txt`, then open either notebook and run it top to bottom.

## Limitations

* The research universe is the three companies with a 10-K in `data/10k/`. To add a company, add its PDF and an entry in `UNIVERSE`.
* Market data uses the free IEX feed, which covers fewer trades than the consolidated SIP feed.
* Orders are market orders. Bracket orders (stop-loss / take-profit) are a natural next step.
* The evaluation set is small and synthetic. It checks behaviour, not investment performance.

## Built with

Pydantic AI · LangChain · Model Context Protocol (Alpaca MCP server v2) · Ollama · NumPy / pandas · pypdf

Started from the final projects of Udacity's *Agentic AI for Finance* Nanodegree (EDGAR-RAG agent and multi-agent trading system).
