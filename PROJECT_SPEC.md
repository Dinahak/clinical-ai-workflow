# Clinical AI Workflow — Family Medicine Decision support — Project Specification

## Overview

This project explores the use of local large language models and retrieval-augmented reasoning to support clinicians in generating differential diagnoses for common chronic diseases.

## 💡 The Problem

Family medicine clinicians are under enormous pressure:

- **High patient load** — GPs see 30–50 patients per day, leaving little time for deep diagnostic reasoning
- **Diagnostic delays** — undifferentiated presentations are easy to miss without structured support
- **Unnecessary testing** — without clear guidance, over-investigation is common and costly
- **Documentation burden** — administrative work consumes up to 40% of a clinician's day

## ✅ What This System Does

This tool acts as a **clinical decision support assistant** running alongside the physician. It does not replace clinical judgement — it handles the cognitive groundwork so the clinician can focus on the patient.

| The clinician inputs... | The system returns... |
|-------------------------|----------------------|
| A consultation note, voice transcript, or EHR export | Structured symptom list with clinical codes |
| Patient age, sex, and history | Ranked differential diagnoses with confidence scores |
| Free-text or structured data | Evidence-backed test recommendations |
| Any case with red-flag features | Immediate urgent alert before anything else |

## Scope and Goals

An AI-powered multi-agent system designed to support family medicine clinicians by reducing burnout, improving diagnostic accuracy, and recommending cost-effective, evidence-based investigations.


### Non-Goals

The system is not intended to:

- Provide a final medical diagnosis
- Recommend medications or treatments
- Provide emergency triage guidance
- Be used directly by patients without clinician oversight

## 🚑 Key Features

- 🔍 **Symptom extraction** — Reads clinical notes and maps symptoms to standardised medical terminology (HPO codes)
- 🧠 **AI differential diagnosis** — Powered by Qwen2.5-72B (local LLM), generates ranked differentials with supporting evidence
- 📚 **Evidence retrieval (RAG)** — Searches a curated knowledge base of NICE guidelines, RCGP protocols, and primary care literature
- ⚠️ **Red-flag detection** — Automatically surfaces cannot-miss diagnoses (sepsis, PE, meningitis, etc.) before other output
- 🧪 **Test recommendations** — Suggests investigations prioritised by clinical yield and cost-effectiveness
- 🧾 **Audit trail** — Every inference step is logged in an append-only store for transparency and clinical governance

  ## 💼 Business & Clinical Value

| Value | Impact |
|-------|--------|
| ⏱️ Saves clinician time | More capacity per session; less after-hours documentation |
| 💰 Reduces unnecessary testing | Targets 15–30% reduction in low-yield investigations |
| 📉 Reduces diagnostic errors | Structured red-flag detection catches what fatigue misses |
| 📊 Improves documentation | Automated audit trail supports quality improvement and medico-legal compliance |
| 🔒 Keeps data local | LLM runs on-device — no patient data leaves the clinic |

  ## High-Level Architecture

The system consists of several core components:

1. Case ingestion service
2. Preprocessing and phenotype extraction
3. Medical knowledge base
4. Retrieval system
5. Local LLM reasoning module
6. Diagnostic agent orchestrator
7. User interface

These components work together to transform clinical case descriptions into a structured phenotype representation and generate a ranked differential diagnosis.
