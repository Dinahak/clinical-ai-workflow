# Clinical AI Workflow — Family Medicine Decision support 
An AI-powered multi-agent system designed to support family medicine clinicians by reducing burnout, improving diagnostic accuracy, and recommending cost-effective, evidence-based investigations.


## Key Features

-  **Symptom extraction** — Reads clinical notes and maps symptoms to standardised medical terminology (HPO codes)
-  **AI differential diagnosis** — Powered by Qwen2.5-72B (local LLM), generates ranked differentials with supporting evidence
-  **Evidence retrieval (RAG)** — Searches a curated knowledge base 
-  **Red-flag detection** — Automatically surfaces cannot-miss diagnoses before other output
-  **Test recommendations** — Suggests investigations prioritised by clinical yield and cost-effectiveness
-  **Audit trail** — Every inference step is logged in an append-only store for transparency and clinical governance

# How It Works

The system runs as three coordinated AI agents:

```
Patient case input
(notes · EHR · audio · HL7/FHIR)
         │
         ▼
  ┌─────────────────────┐
  │  Agent 1            │  Reads the case, extracts symptoms,
  │  Preprocessing      │  maps to clinical codes (HPO/SNOMED)
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  Agent 2            │  Searches medical knowledge base,
  │  Knowledge &        │  retrieves relevant guidelines and
  │  Retrieval          │  evidence for this specific case
  └─────────┬───────────┘
            │
            ▼
  ┌─────────────────────┐
  │  Agent 3            │  Coordinates the full pipeline,
  │  Orchestrator       │  runs self-critique, scores confidence,
  │                     │  writes the audit log
  └─────────┬───────────┘
            │
            ▼
  Safety & validation gate
  (red-flag checks · confidence filter · disclaimer injection)
            │
            ▼
  Physician dashboard
  ┌──────────┬──────────┬──────────────────┐
  │ Urgent   │ Ranked   │ Test             │
  │ alerts   │ diffs    │ recommendations  │
  └──────────┴──────────┴──────────────────┘
```

---
## Intended Users

Clinicians and medical researchers investigating family chronic disease cases.

## Status

Early architecture + research prototype.
