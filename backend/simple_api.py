from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.llm import FeatherlessClient, FeatherlessError
from backend.retrieval import RetrievalStore
from services.parser import get_parser_service

load_dotenv()

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / 'data'
UPLOAD_DIR = DATA_DIR / 'uploads'
INDEX_DIR = DATA_DIR / 'indexes'
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {'.txt', '.pdf', '.json', '.html', '.htm', '.docx', '.xml'}


class IngestTextRequest(BaseModel):
    text: str = Field(min_length=1)
    title: Optional[str] = None


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: Optional[str] = None
    top_k: int = Field(default=6, ge=1, le=12)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    document_id: Optional[str] = None
    top_k: int = Field(default=8, ge=1, le=20)


class MultiAgentRequest(BaseModel):
    question: str = Field(min_length=1)
    document_id: Optional[str] = None
    top_k: int = Field(default=6, ge=1, le=12)
    simulation_overrides: Dict[str, float] = Field(default_factory=dict)


class Analyzer:
    def __init__(self) -> None:
        self.parser = get_parser_service()
        self.store = RetrievalStore(INDEX_DIR)
        self.llm = FeatherlessClient()

    async def ingest_file(self, upload: UploadFile, title: Optional[str] = None) -> Dict[str, Any]:
        suffix = Path(upload.filename or 'upload.txt').suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f'Unsupported file type: {suffix or "unknown"}')
        file_path = UPLOAD_DIR / f'{Path(upload.filename or "upload").stem}-{os.getpid()}{suffix}'
        file_path.write_bytes(await upload.read())
        parsed = await self.parser.parse(str(file_path))
        text = (parsed.text or '').strip()
        if not text:
            raise HTTPException(status_code=422, detail='No readable text could be extracted from the file.')
        return self._persist_document(
            title=title or upload.filename or 'Uploaded document',
            source_type=parsed.format,
            source_name=upload.filename,
            raw_text=text,
            metadata=parsed.metadata,
            tables=parsed.tables,
        )

    async def ingest_text(self, text: str, title: Optional[str] = None) -> Dict[str, Any]:
        clean = text.strip()
        if not clean:
            raise HTTPException(status_code=400, detail='Text input is empty.')
        return self._persist_document(
            title=title or 'Pasted text',
            source_type='text',
            source_name=None,
            raw_text=clean,
            metadata={'characters': len(clean), 'words': len(clean.split())},
            tables=[],
        )

    def _persist_document(
        self,
        *,
        title: str,
        source_type: str,
        source_name: Optional[str],
        raw_text: str,
        metadata: Dict[str, Any],
        tables: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        extracted = {
            'labs': self._extract_labs(raw_text),
            'entities': self._extract_entities(raw_text),
            'tables': tables[:5],
            'sections': self._extract_sections(raw_text),
        }
        ai_summary = self._build_ai_summary(title, raw_text, metadata, extracted)
        return self.store.upsert_document(
            title=title,
            source_type=source_type,
            source_name=source_name,
            raw_text=raw_text,
            metadata={**metadata, 'words': len(raw_text.split()), 'characters': len(raw_text)},
            extracted=extracted,
            ai_summary=ai_summary,
        )

    def _build_ai_summary(self, title: str, raw_text: str, metadata: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if self.llm.configured:
                summary = self.llm.summarize_document(title=title, text=raw_text, metadata=metadata, extracted=extracted)
            else:
                summary = self._fallback_document_summary(title=title, raw_text=raw_text, extracted=extracted)
        except Exception:
            summary = self._fallback_document_summary(title=title, raw_text=raw_text, extracted=extracted)
        summary.setdefault('summary', 'No summary returned.')
        summary.setdefault('bullet_points', [])
        summary.setdefault('entities', extracted.get('entities', {}))
        summary.setdefault('tags', [])
        summary.setdefault('recommendations', [])
        summary.setdefault('risk_level', 'low')
        summary['confidence'] = float(summary.get('confidence', 0.5))
        return summary

    def answer_question(self, question: str, document_id: Optional[str], top_k: int) -> Dict[str, Any]:
        doc = self.store.get_document(document_id) if document_id else None
        grounded_query = self._prepare_grounded_query(question, doc)
        chunks = self.store.search(grounded_query, limit=top_k, document_id=document_id)
        if not chunks:
            return {
                'answer': 'I could not ground that question in the indexed PDF content yet. Try asking about a symptom, lab, diagnosis, medication, or a named section from the document.',
                'citations': [],
                'follow_up_questions': self._suggest_grounded_questions(doc, []),
                'matches': [],
                'grounded_query': grounded_query,
            }
        doc_title = doc['title'] if doc else None
        if self.llm.configured:
            try:
                answer = self.llm.answer_with_context(question=question, context_chunks=chunks, document_title=doc_title)
            except Exception:
                answer = self._fallback_answer(question=question, chunks=chunks, document=doc)
        else:
            answer = self._fallback_answer(question=question, chunks=chunks, document=doc)
        answer['matches'] = chunks
        answer['grounded_query'] = grounded_query
        if not answer.get('citations'):
            answer['citations'] = self._grounded_citations(chunks)
        answer['follow_up_questions'] = self._suggest_grounded_questions(doc, chunks)
        return answer

    def run_multi_agent(self, question: str, document_id: Optional[str], top_k: int, simulation_overrides: Dict[str, float]) -> Dict[str, Any]:
        chunks = self.store.search(question, limit=top_k, document_id=document_id)
        document = self.store.get_document(document_id) if document_id else None
        if not document and chunks:
            document = self.store.get_document(chunks[0]['document_id'])
        if document is None:
            return {
                'workflow': [],
                'final': {
                    'summary': 'No indexed evidence is available yet.',
                    'confidence': 0.0,
                    'recommended_actions': ['Ingest a document and try again.'],
                    'top_diagnoses': [],
                },
            }

        labs = json.loads(json.dumps(document.get('extracted', {}).get('labs', {})))
        for key, value in simulation_overrides.items():
            if key in labs:
                labs[key]['value'] = float(value)
                low = labs[key].get('reference_range', {}).get('min')
                high = labs[key].get('reference_range', {}).get('max')
                if low is not None and value < low:
                    labs[key]['status'] = 'low'
                elif high is not None and value > high:
                    labs[key]['status'] = 'high'
                else:
                    labs[key]['status'] = 'normal'

        entities = document.get('extracted', {}).get('entities', {})
        evidence = self._build_evidence(chunks, labs, entities, question)
        diagnoses = self._score_diagnoses(labs, entities, question)
        validation = self._validate_diagnoses(diagnoses, labs)
        critique = self._critique_diagnoses(diagnoses, labs, entities)
        final = self._compose_final_response(question, document, diagnoses, validation, critique, evidence, simulation_overrides)

        workflow = [
            {
                'agent': 'ingestion-agent',
                'title': 'Ingestion Agent',
                'status': 'completed',
                'summary': 'Document already indexed and structured.',
                'details': {
                    'document': document['title'],
                    'source_type': document['source_type'],
                    'chunk_count': len(document.get('chunks', [])),
                    'available_labs': list(labs.keys()),
                },
            },
            {
                'agent': 'retrieval-agent',
                'title': 'Retrieval Agent',
                'status': 'completed',
                'summary': f'Retrieved {len(chunks)} evidence chunk(s) relevant to the question.',
                'details': {
                    'question': question,
                    'matches': [
                        {
                            'chunk_id': item['chunk_id'],
                            'score': round(float(item.get('score', 0.0)), 4),
                            'preview': item['text'][:240],
                        }
                        for item in chunks[:5]
                    ],
                },
            },
            {
                'agent': 'diagnosis-agent',
                'title': 'Diagnosis Agent',
                'status': 'completed',
                'summary': 'Generated ranked hypotheses from structured evidence.',
                'details': {'top_diagnoses': diagnoses},
            },
            {
                'agent': 'validation-agent',
                'title': 'Validation Agent',
                'status': 'completed',
                'summary': 'Checked diagnosis support against lab thresholds and symptoms.',
                'details': validation,
            },
            {
                'agent': 'critic-agent',
                'title': 'Critic Agent',
                'status': 'completed',
                'summary': 'Challenged the leading diagnosis and surfaced uncertainty.',
                'details': critique,
            },
            {
                'agent': 'explanation-agent',
                'title': 'Explanation Agent',
                'status': 'completed',
                'summary': 'Translated agent outputs into a presentation-ready response.',
                'details': final,
            },
        ]
        return {
            'workflow': workflow,
            'final': final,
            'evidence': evidence,
            'simulation': {'applied_overrides': simulation_overrides, 'labs': labs},
        }

    def semantic_search(self, query: str, document_id: Optional[str], top_k: int) -> Dict[str, Any]:
        document = self.store.get_document(document_id) if document_id else None
        grounded_query = self._prepare_grounded_query(query, document)
        items = self.store.search(grounded_query, limit=top_k, document_id=document_id)
        suggestions = self._suggest_grounded_questions(document, items)
        return {
            'items': items,
            'grounded_query': grounded_query,
            'suggested_questions': suggestions,
        }

    def _prepare_grounded_query(self, question: str, document: Optional[Dict[str, Any]]) -> str:
        question = re.sub(r'\s+', ' ', (question or '').strip())
        if not document:
            return question

        entities = document.get('extracted', {}).get('entities', {}) or {}
        labs = document.get('extracted', {}).get('labs', {}) or {}
        sections = document.get('extracted', {}).get('sections', []) or []

        boosters: List[str] = []
        boosters.append(document.get('title', ''))
        for bucket in ('conditions', 'symptoms', 'medications'):
            boosters.extend(entities.get(bucket, [])[:4])
        for lab_name, payload in list(labs.items())[:5]:
            status = payload.get('status')
            if status and status != 'normal':
                boosters.append(f'{lab_name} {status}')
                boosters.append(lab_name)
        for section in sections[:3]:
            heading = (section.get('heading') or '').strip()
            if heading:
                boosters.append(heading)

        deduped: List[str] = []
        seen = set()
        for item in boosters:
            clean = re.sub(r'[^a-zA-Z0-9 /_-]+', ' ', str(item)).strip().lower()
            if len(clean) < 3 or clean in seen:
                continue
            seen.add(clean)
            deduped.append(clean)

        extra = ' '.join(deduped[:8])
        return f"{question} {extra}".strip()

    def _grounded_citations(self, chunks: List[Dict[str, Any]], limit: int = 3) -> List[str]:
        cites = []
        for idx, chunk in enumerate(chunks[:limit]):
            snippet = re.sub(r'\s+', ' ', chunk.get('text', '')).strip()[:140]
            cites.append(f"Chunk {idx + 1} · {snippet}")
        return cites

    def _suggest_grounded_questions(self, document: Optional[Dict[str, Any]], chunks: List[Dict[str, Any]]) -> List[str]:
        suggestions: List[str] = []
        entities = (document or {}).get('extracted', {}).get('entities', {}) or {}
        labs = (document or {}).get('extracted', {}).get('labs', {}) or {}
        sections = (document or {}).get('extracted', {}).get('sections', []) or []

        abnormal_labs = [name for name, payload in labs.items() if payload.get('status') in {'high', 'low'}]
        if abnormal_labs:
            joined = ', '.join(abnormal_labs[:2])
            suggestions.append(f'Which parts of the PDF explain the abnormal findings for {joined}?')
            suggestions.append(f'What clinical risk is suggested by the {joined} values in this document?')

        symptoms = entities.get('symptoms', [])[:3]
        if symptoms:
            suggestions.append(f'How do the documented symptoms ({", ".join(symptoms)}) connect to the likely diagnosis?')

        conditions = entities.get('conditions', [])[:3]
        if conditions:
            suggestions.append(f'What evidence in the PDF supports or weakens {conditions[0]}?')

        for section in sections[:2]:
            heading = (section.get('heading') or '').strip()
            if heading and heading.lower() != 'extracted text':
                suggestions.append(f'What are the key takeaways from the {heading} section?')

        if chunks:
            suggestions.append('Which retrieved chunk is the strongest evidence, and why?')
            suggestions.append('What important information is still missing from the document to answer this confidently?')

        out: List[str] = []
        seen = set()
        for item in suggestions:
            norm = item.lower()
            if norm in seen:
                continue
            seen.add(norm)
            out.append(item)
        return out[:6]

    def _extract_labs(self, text: str) -> Dict[str, Dict[str, Any]]:
        patterns = {
            'hemoglobin': r'(?:hemoglobin|hb)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(g/dl|gdl)?',
            'wbc': r'(?:wbc|white blood cell(?: count)?)\s*[:=]?\s*(\d+(?:\.\d+)?)',
            'platelets': r'(?:platelets?|plt)\s*[:=]?\s*(\d+(?:\.\d+)?)',
            'creatinine': r'(?:creatinine)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dl)?',
            'glucose': r'(?:glucose|blood sugar)\s*[:=]?\s*(\d+(?:\.\d+)?)\s*(mg/dl)?',
            'sodium': r'(?:sodium|na)\s*[:=]?\s*(\d+(?:\.\d+)?)',
            'potassium': r'(?:potassium|k)\s*[:=]?\s*(\d+(?:\.\d+)?)',
        }
        ranges = {
            'hemoglobin': (12.0, 17.5),
            'wbc': (4.0, 11.0),
            'platelets': (150, 450),
            'creatinine': (0.6, 1.3),
            'glucose': (70, 140),
            'sodium': (135, 145),
            'potassium': (3.5, 5.1),
        }
        out: Dict[str, Dict[str, Any]] = {}
        lower = text.lower()
        for name, pattern in patterns.items():
            match = re.search(pattern, lower, flags=re.I)
            if not match:
                continue
            value = float(match.group(1))
            low, high = ranges[name]
            status = 'normal'
            if value < low:
                status = 'low'
            elif value > high:
                status = 'high'
            out[name] = {
                'value': value,
                'unit': (match.group(2) or '').strip() if len(match.groups()) > 1 else '',
                'reference_range': {'min': low, 'max': high},
                'status': status,
            }
        return out

    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        lower = text.lower()
        buckets = {
            'symptoms': ['fever', 'cough', 'fatigue', 'dyspnea', 'chest pain', 'headache', 'vomiting', 'nausea', 'dizziness', 'pale skin', 'tiredness'],
            'conditions': ['diabetes', 'hypertension', 'anemia', 'sepsis', 'infection', 'kidney disease', 'asthma', 'covid', 'pneumonia'],
            'medications': ['metformin', 'insulin', 'amlodipine', 'paracetamol', 'acetaminophen', 'ibuprofen', 'aspirin', 'lisinopril'],
        }
        return {bucket: [term for term in terms if term in lower] for bucket, terms in buckets.items()}

    def _extract_sections(self, text: str) -> List[Dict[str, str]]:
        section_names = ['chief complaint', 'history', 'impression', 'assessment', 'plan', 'diagnosis', 'medications', 'recommendations']
        splitter = re.compile(r'(?i)(' + '|'.join(re.escape(name) for name in section_names) + r')\s*:')
        if not splitter.search(text):
            return [{'heading': 'Extracted text', 'content': text[:2500]}]
        pieces = splitter.split(text)
        sections: List[Dict[str, str]] = []
        for idx in range(1, len(pieces), 2):
            heading = pieces[idx].strip().title()
            content = pieces[idx + 1].strip() if idx + 1 < len(pieces) else ''
            if content:
                sections.append({'heading': heading, 'content': content[:1800]})
        return sections or [{'heading': 'Extracted text', 'content': text[:2500]}]

    def _fallback_document_summary(self, title: str, raw_text: str, extracted: Dict[str, Any]) -> Dict[str, Any]:
        labs = extracted.get('labs', {})
        entities = extracted.get('entities', {})
        abnormal = [f"{name}: {item['value']} ({item['status']})" for name, item in labs.items() if item.get('status') != 'normal']
        symptoms = entities.get('symptoms', [])
        tags = list(dict.fromkeys([*symptoms[:3], *[name for name in labs.keys()], *entities.get('conditions', [])[:2]]))[:6]
        risk = 'low'
        if any(item.get('status') == 'low' for item in labs.values()) or 'anemia' in entities.get('conditions', []):
            risk = 'moderate'
        if any(name == 'hemoglobin' and item.get('value', 99) < 8 for name, item in labs.items()):
            risk = 'high'
        bullets = abnormal[:4] or [raw_text[:180]]
        return {
            'summary': f'{title} was analyzed with offline heuristics. Key signals were extracted for search and question answering.',
            'bullet_points': bullets,
            'entities': entities,
            'tags': tags,
            'recommendations': ['Review extracted evidence.', 'Use the multi-agent view to inspect how each step contributed.'],
            'risk_level': risk,
            'confidence': 0.68,
        }

    def _fallback_answer(self, question: str, chunks: List[Dict[str, Any]], document: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        evidence = chunks[:3]
        labs = document.get('extracted', {}).get('labs', {}) if document else {}
        entities = document.get('extracted', {}).get('entities', {}) if document else {}
        findings = []
        for name, item in labs.items():
            findings.append(f"{name} {item.get('value')} ({item.get('status')})")
        if entities.get('symptoms'):
            findings.append('symptoms: ' + ', '.join(entities['symptoms']))
        summary = ' '.join(findings[:4]) or 'Relevant evidence was found in the retrieved chunks.'
        return {
            'answer': f"Heuristic answer based on retrieved evidence: {summary}",
            'citations': self._grounded_citations(evidence),
            'follow_up_questions': self._suggest_grounded_questions(document, evidence),
        }

    def _build_evidence(self, chunks: List[Dict[str, Any]], labs: Dict[str, Any], entities: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        cards: List[Dict[str, Any]] = []
        for name, item in labs.items():
            cards.append({
                'type': 'lab',
                'label': name,
                'value': item.get('value'),
                'status': item.get('status'),
                'reason': f"Reference range {item.get('reference_range', {}).get('min')}–{item.get('reference_range', {}).get('max')}"
            })
        for symptom in entities.get('symptoms', [])[:4]:
            cards.append({'type': 'symptom', 'label': symptom, 'value': 'present', 'status': 'supporting', 'reason': 'Detected in extracted entities'})
        for idx, chunk in enumerate(chunks[:3]):
            cards.append({'type': 'chunk', 'label': f'Evidence chunk {idx + 1}', 'value': round(float(chunk.get('score', 0.0)), 3), 'status': 'retrieved', 'reason': chunk.get('text', '')[:160]})
        return cards

    def _score_diagnoses(self, labs: Dict[str, Any], entities: Dict[str, Any], question: str) -> List[Dict[str, Any]]:
        symptoms = set(entities.get('symptoms', []))
        conditions = set(entities.get('conditions', []))
        hb = labs.get('hemoglobin', {}).get('value')
        wbc = labs.get('wbc', {}).get('value')
        platelets = labs.get('platelets', {}).get('value')
        dx: List[Dict[str, Any]] = []

        if hb is not None:
            score = 0.55
            reasons = [f'Hemoglobin is {hb} g/dL']
            if hb < 12:
                score += 0.2
                reasons.append('Below the reference threshold for anemia')
            if hb < 9:
                score += 0.1
                reasons.append('Severity is more than mild')
            if {'dizziness', 'fatigue', 'tiredness', 'pale skin'} & symptoms:
                score += 0.08
                reasons.append('Symptoms align with reduced oxygen carrying capacity')
            if 'anemia' in conditions or 'diagnosis' in question.lower():
                score += 0.05
            dx.append({'name': 'Anemia', 'confidence': min(score, 0.96), 'rationale': reasons})
            dx.append({'name': 'Iron Deficiency Anemia', 'confidence': min(max(score - 0.07, 0.28), 0.89), 'rationale': ['Low hemoglobin supports this possibility', 'Needs iron studies to confirm']})
        if wbc is not None and wbc > 11:
            dx.append({'name': 'Infection / Inflammatory Process', 'confidence': 0.58, 'rationale': [f'WBC is elevated at {wbc}']})
        if platelets is not None and platelets < 150:
            dx.append({'name': 'Thrombocytopenia', 'confidence': 0.61, 'rationale': [f'Platelets are low at {platelets}']})
        if not dx:
            dx.append({'name': 'Insufficient evidence', 'confidence': 0.32, 'rationale': ['No strong disease-specific signal was found in extracted labs.']})

        dx.sort(key=lambda item: item['confidence'], reverse=True)
        return dx[:3]

    def _validate_diagnoses(self, diagnoses: List[Dict[str, Any]], labs: Dict[str, Any]) -> Dict[str, Any]:
        checks = []
        hb = labs.get('hemoglobin')
        if hb:
            supported = hb.get('value', 99) < hb.get('reference_range', {}).get('min', 12)
            checks.append({'rule': 'hemoglobin-below-range', 'passed': supported, 'detail': f"Hb={hb.get('value')}"})
        normal_labs = [name for name, item in labs.items() if item.get('status') == 'normal']
        return {
            'checks': checks,
            'normal_labs': normal_labs,
            'support_strength': 'strong' if any(item['passed'] for item in checks) else 'moderate',
        }

    def _critique_diagnoses(self, diagnoses: List[Dict[str, Any]], labs: Dict[str, Any], entities: Dict[str, Any]) -> Dict[str, Any]:
        concerns = []
        if diagnoses and diagnoses[0]['name'] == 'Anemia':
            concerns.append('Primary diagnosis is broad; etiology still needs confirmation.')
            concerns.append('Ferritin, MCV, and reticulocyte count would help separate iron deficiency from other causes.')
        if not entities.get('medications'):
            concerns.append('No medication context was extracted, so treatment interactions were not assessed.')
        if not labs.get('hemoglobin'):
            concerns.append('No hemoglobin value was extracted, which weakens diagnostic confidence.')
        return {
            'counterpoints': concerns,
            'missing_data': ['Ferritin', 'MCV', 'Reticulocyte count'] if labs.get('hemoglobin') else ['Core lab values'],
        }

    def _compose_final_response(self, question: str, document: Dict[str, Any], diagnoses: List[Dict[str, Any]], validation: Dict[str, Any], critique: Dict[str, Any], evidence: List[Dict[str, Any]], simulation_overrides: Dict[str, float]) -> Dict[str, Any]:
        best = diagnoses[0]
        summary = f"Top finding: {best['name']} with {round(best['confidence'] * 100)}% confidence."
        if simulation_overrides:
            summary += ' Simulation mode adjusted the lab inputs before scoring.'
        return {
            'summary': summary,
            'confidence': best['confidence'],
            'top_diagnoses': diagnoses,
            'recommended_actions': [
                'Review the evidence cards and supporting chunks.',
                'Use the critic notes to explain uncertainty in your demo.',
                'Expose /api/agents/sync and /api/ask in the pitch as separate layers of the system.',
            ],
            'critique': critique,
            'validation': validation,
            'document_title': document['title'],
            'question': question,
            'evidence_count': len(evidence),
        }


analyzer = Analyzer()
app = FastAPI(title='DeepSeek Featherless RAG API', version='3.0.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/api/health')
def health() -> Dict[str, Any]:
    docs = analyzer.store.list_documents(limit=200)
    return {
        'status': 'ok',
        'featherless': {
            'configured': analyzer.llm.configured,
            'base_url': os.getenv('FEATHERLESS_BASE_URL', 'https://api.featherless.ai/v1'),
            'model': os.getenv('FEATHERLESS_MODEL', 'Qwen/Qwen2.5-7B-Instruct'),
            'mode': 'live-llm' if analyzer.llm.configured else 'offline-heuristic-fallback',
        },
        'storage': {'documents': len(docs), 'upload_dir': str(UPLOAD_DIR), 'index_dir': str(INDEX_DIR)},
        'features': [
            'pdf-text-ingestion',
            'semantic-search',
            'grounded-qa',
            'multi-agent-sync',
            'simulation-mode',
            'api-demo-panel',
        ],
    }


@app.get('/api/documents')
def list_documents(limit: int = 50) -> Dict[str, Any]:
    return {'items': analyzer.store.list_documents(limit=limit)}


@app.get('/api/documents/{document_id}')
def get_document(document_id: str) -> Dict[str, Any]:
    document = analyzer.store.get_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail='Document not found')
    document['suggested_questions'] = analyzer._suggest_grounded_questions(document, [])
    return document


@app.post('/api/ingest/text')
async def ingest_text(payload: IngestTextRequest) -> Dict[str, Any]:
    try:
        return await analyzer.ingest_text(payload.text, payload.title)
    except FeatherlessError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/api/ingest/file')
async def ingest_file(file: UploadFile = File(...), title: Optional[str] = Form(None)) -> Dict[str, Any]:
    try:
        return await analyzer.ingest_file(file, title)
    except FeatherlessError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/api/search')
def search(payload: SearchRequest) -> Dict[str, Any]:
    return analyzer.semantic_search(payload.query, payload.document_id, payload.top_k)


@app.post('/api/ask')
def ask(payload: AskRequest) -> Dict[str, Any]:
    try:
        return analyzer.answer_question(payload.question, payload.document_id, payload.top_k)
    except FeatherlessError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post('/api/agents/sync')
def agents_sync(payload: MultiAgentRequest) -> Dict[str, Any]:
    return analyzer.run_multi_agent(payload.question, payload.document_id, payload.top_k, payload.simulation_overrides)
