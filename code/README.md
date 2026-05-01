# Orchestrate Prototype

This `code/` directory contains the phase-one prototype for the HackerRank Orchestrate challenge.

## What is implemented

- Rule-based guard agent for prompt injection, adversarial requests, and obvious sensitive-number detection
- Rule-based triage agent for domain routing, request classification, urgency, and escalation hints
- Hybrid retriever that combines BM25 lexical search with dense semantic search in Chroma using `all-MiniLM-L6-v2`
- Response synthesizer with two modes:
  - Template mode with no API dependency
  - Optional LLM mode using environment variables
- Escalation judge for final safety and confidence checks
- Terminal entry point that reads `../support_tickets/support_tickets.csv` and writes `../support_tickets/output.csv`
- Interactive CLI mode for live ticket testing
- Textual TUI mode with streaming Anthropic responses
- Guided Textual workflow with company selection, subject, multiline issue entry, CSV ingestion, similar-incident retrieval, structured state output, references, and confidence display

## Optional LLM providers

The pipeline runs without an LLM, but you can enable one for better grounded synthesis.

### Groq

Groq support is wired for the OpenAI-compatible API using the official Groq endpoint and the current 70B Llama model id:

- Provider: `groq`
- Default model: `llama-3.3-70b-versatile`
- API key env var: `GROQ_API_KEY`

Example PowerShell setup:

```powershell
$env:ORCHESTRATE_LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-key"
$env:ORCHESTRATE_LLM_MODEL = "llama-3.3-70b-versatile"
python .\code\main.py
```

### OpenAI

```powershell
$env:ORCHESTRATE_LLM_PROVIDER = "openai"
$env:OPENAI_API_KEY = "your-key"
python .\code\main.py
```

### Anthropic

```powershell
$env:ORCHESTRATE_LLM_PROVIDER = "anthropic"
$env:ANTHROPIC_API_KEY = "your-key"
python .\code\main.py
```

### Textual TUI streaming

The Textual app is provider-aware:

- `anthropic`: uses Anthropic's async streaming client with `client.messages.stream(...)`
- `groq`: uses Groq's OpenAI-compatible streaming endpoint through the async OpenAI client
- `openai`: uses the async OpenAI client directly

```powershell
$env:ORCHESTRATE_LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-key"
$env:ORCHESTRATE_TUI_MODEL = "llama-3.3-70b-versatile"
python .\code\main.py --mode tui
```

Inside the TUI:

- Select `Company`
- Enter `Subject`
- Enter multiline `Issue`
- Optionally set a `.csv` path and click `Ingest CSV`
- Click `Retrieve Similar` to get semantically similar incidents from `sample_support_tickets.csv`
- Click `AI Recommendation` to get a grounded recommendation with:
  - `status`
  - `product_area`
  - `request_type`
  - `risk_level`
  - `confidence`
  - streamed response
  - retrieved references

## Run

From the repo root:

```powershell
python .\code\indexer.py
python .\code\main.py
python .\code\main.py --mode interactive
python .\code\main.py --mode tui
```

## Notes

- All application code lives under `code/`, per the repo contract.
- Retrieval now uses BM25 + Chroma dense search + Reciprocal Rank Fusion.
- Chroma's official docs support dense, sparse, and hybrid retrieval, so we stayed on Chroma instead of moving to Qdrant.
- The agent uses only the local support corpus under `data/`.
- The Textual app is fully async and keeps API work off the UI thread.
