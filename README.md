# Orchestrate Prototype

This `code/` directory contains the phase-one prototype for the HackerRank Orchestrate challenge.

## What is implemented

- Rule-based guard agent for prompt injection, adversarial requests, and obvious sensitive-number detection
- Rule-based triage agent for domain routing, request classification, urgency, and escalation hints
- **Knowledge graph** (`graph_store.py`) that maps corpus sections to support concepts and expands retrieval to related sections at query time
- Hybrid retriever that combines BM25 lexical search with dense semantic search in Chroma using `all-MiniLM-L6-v2`, fused via Reciprocal Rank Fusion, then graph-expanded into evidence bundles
- Response synthesizer with two modes:
  - Template mode with no API dependency
  - Optional LLM mode using environment variables
- Hallucination verifier that checks generated responses against retrieved evidence and escalates high-risk unsupported answers
- Escalation judge for final safety and confidence checks
- Terminal entry point that reads `../support_tickets/support_tickets.csv` and writes `../support_tickets/output.csv`
- Rich-powered batch progress display with live stage updates
- Knowledge-gap report under `../log/knowledge_gaps_latest.json`
- Interactive CLI mode for live ticket testing
- Textual TUI mode with async streaming-style updates from the final verified pipeline response
- Guided Textual workflow with company selection, subject, multiline issue entry, CSV ingestion, similar-incident retrieval, structured state output, references, confidence display, and saved-incident memory

## Knowledge Graph

The knowledge graph (`graph_store.py`) sits between the hybrid retriever and the response agent. Rather than returning a flat ranked list of chunks, the retriever groups chunks by **corpus section** and then uses the graph to pull in *related* sections that share support concepts — giving the response agent richer, more complete evidence bundles.

### How it works

**1. Concept ontology (`SUPPORT_CONCEPTS`)**

Thirteen top-level support concepts are defined, each with a list of trigger phrases:

| Concept | Example triggers |
|---|---|
| `refund` | refund, reimburse, money back |
| `billing` | billing, invoice, payment, subscription |
| `mock_interview` | mock interview, interview credits |
| `candidate_assessment` | assessment, candidate, score, reinvite |
| `account_access` | access, login, admin, workspace |
| `user_management` | deactivate user, team member, employee |
| `security_review` | infosec, security questionnaire, vendor |
| `privacy` | privacy, delete, export, data retention |
| `fraud` | fraud, identity theft, unauthorized |
| `card_support` | blocked card, lost card, stolen card |
| `merchant_dispute` | merchant, dispute, chargeback |
| `contact` | contact, email, phone, support team |
| `policy` | policy, rules, must, cannot |

**2. Section–concept index**

At startup, `SupportGraphStore` scans every corpus section (title + body text) and records which concepts each section covers. An inverted index (`concept_sections`) is built so the graph can jump instantly from a concept name to all corpus sections that discuss it.

**3. Graph expansion at retrieval time**

For each primary chunk returned by BM25 + dense retrieval the graph calls `related_section_ids(primary, query_terms, target_terms)`:

- Collects concepts from the **query text** and from the **primary chunk's section**.
- Walks the inverted index to find candidate sections that share those concepts **and** belong to the same product domain (HackerRank / Claude / Visa).
- Scores each candidate: `score = 0.006 × target_term_overlap + 0.004 × concept_overlap`
- Returns up to 4 sibling sections, each annotated with its traversal path — e.g. `faq-billing → refund → faq-refund-policy`.

**4. Evidence bundles**

The retriever merges primary chunks and graph-expanded chunks into a single ranked evidence bundle passed to the response agent. This means a question about "refund" automatically pulls in billing policy, subscription terms, and any other sections the graph links — without the user needing to mention those exact words.

### Inspecting graph paths

Each retrieved chunk in the trace log (`log/ticket_trace_latest.json`) includes a `graph_path` field showing the concept hop used to include that section, for example:

```
faq-billing-general → billing → faq-subscription-refunds
```

This makes it straightforward to audit which concept edges contributed evidence to any given response.

## Optional LLM providers

The pipeline runs without an LLM, but you can enable one for better grounded synthesis.

### Groq

- Provider: `groq`
- Default model: `llama-3.3-70b-versatile`
- API key env var: `GROQ_API_KEY`

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

### Textual TUI

The Textual app is provider-aware:

- `anthropic`: uses Anthropic's async streaming client with `client.messages.stream(...)`
- `groq`: uses Groq's OpenAI-compatible streaming endpoint through the async OpenAI client
- `openai`: uses the async OpenAI client directly

By default, the TUI streams the pipeline's final verified response rather than asking the model to rewrite the answer after validation. This keeps the TUI aligned with `output.csv`. Set `ORCHESTRATE_TUI_REWRITE=1` to force an LLM rewrite in the TUI.

```powershell
$env:ORCHESTRATE_LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-key"
$env:ORCHESTRATE_TUI_MODEL = "llama-3.3-70b-versatile"
python .\code\main.py --mode tui
```

Inside the TUI:

- Select `Company`
- Enter `Subject` (optional)
- Enter multiline `Issue`
- Optionally set a `.csv` path and click `Ingest CSV`
- Click `Retrieve Similar` to get semantically similar incidents from `sample_support_tickets.csv` plus locally saved incidents
- Click `AI Recommendation` to get a grounded recommendation with:
  - `status`, `product_area`, `request_type`, `risk_level`, `confidence`
  - streamed response text
  - retrieved references (including graph-expanded sections)
- Click `Save Incident` after a recommendation to append the incident and its final resolution to `support_tickets/saved_incidents.csv`

The TUI uses a single-result workspace rather than a chat transcript. Running a new incident replaces the previous result panel.

Saved incidents:

- Are used immediately by semantic similar-incident retrieval in the same TUI session.
- Are stored in `support_tickets/saved_incidents.csv`.
- Are ignored by Git as local operator memory.

## Run

From the repo root:

```powershell
python .\code\indexer.py
python .\code\main.py
python .\code\main.py --mode interactive
python .\code\main.py --mode tui
```

Batch mode writes:

- `support_tickets/output.csv` — evaluator-facing predictions
- `log/ticket_trace_latest.json` — per-ticket stage / retrieval / hallucination / graph-path traces
- `log/knowledge_gaps_latest.json` — low-confidence or weakly grounded tickets that may need more corpus coverage

## Notes

- All application code lives under `code/`, per the repo contract.
- Retrieval uses BM25 + Chroma dense search + Reciprocal Rank Fusion + knowledge-graph expansion.
- Chroma's official docs support dense, sparse, and hybrid retrieval, so we stayed on Chroma instead of moving to Qdrant.
- The agent uses only the local support corpus under `data/`.
- The Textual app is fully async and keeps API work off the UI thread.
