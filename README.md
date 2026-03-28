# Clinical Intelligence Copilot — Multi-Agent Featherless RAG

This version keeps the original logic intact:

1. ingest documents
2. index chunks
3. run semantic search
4. ask grounded questions

It adds the hackathon layer on top:

- visible agent orchestration
- simulation mode for lab values
- evidence cards for explainability
- API demo panel for platform storytelling
- offline fallback mode so the app still works without a Featherless key

## What changed

### Frontend
- redesigned as a hackathon demo dashboard
- added agent workflow timeline
- added simulation controls for hemoglobin, WBC, and platelets
- added API demo panel with ready-to-use curl examples
- preserved document ingestion, semantic search, grounded QA, and document inspection

### Backend
- added `POST /api/agents/sync`
- added deterministic multi-agent orchestration output:
  - ingestion agent
  - retrieval agent
  - diagnosis agent
  - validation agent
  - critic agent
  - explanation agent
- added offline summary and QA fallbacks when `FEATHERLESS_API_KEY` is missing
- preserved the original endpoints:
  - `GET /api/health`
  - `GET /api/documents`
  - `GET /api/documents/{document_id}`
  - `POST /api/ingest/text`
  - `POST /api/ingest/file`
  - `POST /api/search`
  - `POST /api/ask`

## Run locally

### Backend
```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## Environment

Optional for live LLM mode:

```bash
FEATHERLESS_API_KEY=your_key_here
FEATHERLESS_MODEL=Qwen/Qwen2.5-7B-Instruct
FEATHERLESS_BASE_URL=https://api.featherless.ai/v1
```

Without a key, the app uses offline heuristic fallback logic so the demo still runs.

## Hackathon pitch angle

> We did not just build a RAG app. We built a clinical intelligence copilot where specialized agents retrieve evidence, propose a diagnosis, validate it, challenge it, and then explain it — all while exposing APIs that make the whole system reusable as a platform.
