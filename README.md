# AI Support Triage System

An AI-powered support triage and response generation system designed to categorize, analyze, and draft grounded responses for incoming support tickets. The system features a multi-agent validation pipeline, hybrid retrieval (combining lexical and semantic search), graph-based document expansion, and an interactive Textual-based terminal user interface (TUI).

## What is Implemented

- **Guard Agent**: A rule-based validation step to detect prompt injection, adversarial requests, and sensitive data leakage (e.g. social security or credit card numbers) before processing.
- **Triage Agent**: A rule-based routing component that classifies tickets by domain, categories, urgency levels, and identifies potential escalation triggers.
- **Knowledge Graph (`graph_store.py`)**: A graph database representation mapping corpus sections to support concepts. It dynamically expands document retrieval by pulling related sections at query time.
- **Hybrid Retriever**: Combines BM25 lexical search with dense semantic search (using Chroma DB and `all-MiniLM-L6-v2` embeddings). Results are fused using Reciprocal Rank Fusion (RRF) and expanded using the knowledge graph to build rich evidence bundles.
- **Response Synthesizer**: Supports two execution modes:
  - **Template Mode**: Fallback rule-based templates with zero external API dependencies.
  - **LLM Mode**: Grounded response generation using LLM providers via environment configurations.
- **Hallucination Verifier**: Automatically cross-references generated responses against retrieved evidence to flag and escalate unsupported statements.
- **Escalation Judge**: Performs a final safety, confidence, and compliance check before outputting results.
- **Batch Processing CLI**: Reads support tickets from `support_tickets/support_tickets.csv` and outputs predictions/drafts to `support_tickets/output.csv` with a Rich-powered progress interface.
- **Interactive CLI**: Allows live ticket testing directly in the console.
- **Interactive TUI**: Built with Textual, featuring async streaming updates, incident history lookups, multiline input, and similarity comparison.

## Knowledge Graph

The knowledge graph (`graph_store.py`) enhances the retrieval step. Instead of relying solely on flat keyword/vector matches, the hybrid retriever groups chunks by corpus sections and queries the graph to retrieve sibling or related sections sharing identical support concepts. This delivers complete and context-rich evidence bundles to the response generator.

### How it Works

1. **Concept Ontology (`SUPPORT_CONCEPTS`)**
   Thirteen core support concepts are defined with associated trigger phrase lists (e.g., `refund`, `billing`, `account_access`, `user_management`, `privacy`, `fraud`).

2. **Section–Concept Inverted Index**
   At startup, the system scans the corpus documents, mapping each section's contents to relevant support concepts, creating an inverted index for instant conceptual lookups.

3. **Graph Expansion**
   For primary chunks returned by the hybrid retriever, the system calls `related_section_ids()` to:
   - Identify concepts in both the user's query and the primary chunk's parent document.
   - Walk the inverted index to find sibling documents sharing those concepts under the same product domain.
   - Score and select up to 4 related documents using a weighted formula: `score = 0.006 * target_term_overlap + 0.004 * concept_overlap`.

4. **Evidence Bundling**
   Both primary and graph-expanded documents are merged and ranked, ensuring that queries about billing issues automatically retrieve refund policies or payment methods even if those exact terms were not searched.

### Auditing Graph Paths
Every processed ticket produces a trace log (`log/ticket_trace_latest.json`) mapping the concept hops used during retrieval:
```
faq-billing-general → billing → faq-subscription-refunds
```

## Optional LLM Providers

The pipeline can run offline without an LLM. To enable LLM-based response synthesis, configure one of the following providers:

### Groq
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

## Interactive TUI

The Textual TUI app is fully asynchronous and supports streaming responses:
- `anthropic`: Streams via the official Anthropic client.
- `groq` / `openai`: Streams via the OpenAI client interface.

By default, the TUI streams the pipeline's final verified output to ensure alignment with batch predictions. Set `ORCHESTRATE_TUI_REWRITE=1` to allow the LLM to rewrite responses in the UI.

To launch the TUI:
```powershell
$env:ORCHESTRATE_LLM_PROVIDER = "groq"
$env:GROQ_API_KEY = "your-key"
$env:ORCHESTRATE_TUI_MODEL = "llama-3.3-70b-versatile"
python .\code\main.py --mode tui
```

### Key Features of the TUI:
- Select product/company contexts.
- Enter support query subjects and descriptions.
- Load historical tickets from custom CSV files.
- Query similar past incidents using semantic search.
- Stream AI-generated triage classifications and drafts.
- Save resolved tickets to `support_tickets/saved_incidents.csv` for local similarity caching.

## How to Run

Run commands from the repository root:

```powershell
# Index the document corpus
python .\code\indexer.py

# Process support tickets in batch mode
python .\code\main.py

# Launch interactive CLI
python .\code\main.py --mode interactive

# Launch interactive TUI
python .\code\main.py --mode tui
```

### Output Files
- `support_tickets/output.csv`: Batch triage and response predictions.
- `log/ticket_trace_latest.json`: Detailed execution trace showing agent states, retrieved files, and graph path hops.
- `log/knowledge_gaps_latest.json`: Structured list of tickets resolving with low confidence, highlighting document corpus gaps.

## Development Details
- **Architecture**: All core application code is located in the `code/` directory.
- **Storage**: Retrieval uses BM25 combined with Chroma DB for vector storage.
- **Data Boundary**: The agent queries only the local support corpus under `data/` to keep responses strictly grounded.
- **Async Execution**: The Textual UI runs on a separate thread from API network calls to keep the interface highly responsive.

