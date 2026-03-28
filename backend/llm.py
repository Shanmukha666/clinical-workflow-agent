from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import requests

FEATHERLESS_BASE_URL = os.getenv('FEATHERLESS_BASE_URL', 'https://api.featherless.ai/v1').rstrip('/')
FEATHERLESS_MODEL = os.getenv('FEATHERLESS_MODEL', 'Qwen/Qwen2.5-7B-Instruct')
FEATHERLESS_TIMEOUT = float(os.getenv('FEATHERLESS_TIMEOUT', '90'))


class FeatherlessError(RuntimeError):
    pass


class FeatherlessClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or os.getenv('FEATHERLESS_API_KEY', '').strip()
        self.base_url = (base_url or FEATHERLESS_BASE_URL).rstrip('/')
        self.model = model or FEATHERLESS_MODEL

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def require(self) -> None:
        if not self.configured:
            raise FeatherlessError('FEATHERLESS_API_KEY is required. Add it to your .env before starting the backend.')

    def chat(self, messages: List[Dict[str, Any]], *, temperature: float = 0.2, max_tokens: int = 1200, response_format: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.require()
        payload: Dict[str, Any] = {
            'model': self.model,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens,
        }
        if response_format:
            payload['response_format'] = response_format

        response = requests.post(
            f'{self.base_url}/chat/completions',
            headers={
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.api_key}',
            },
            json=payload,
            timeout=FEATHERLESS_TIMEOUT,
        )
        if not response.ok:
            raise FeatherlessError(f'Featherless request failed ({response.status_code}): {response.text[:500]}')
        data = response.json()
        try:
            content = data['choices'][0]['message']['content']
        except Exception as exc:
            raise FeatherlessError(f'Unexpected Featherless response: {data}') from exc
        return {'raw': data, 'content': content}

    def summarize_document(self, *, title: str, text: str, metadata: Dict[str, Any], extracted: Dict[str, Any]) -> Dict[str, Any]:
        prompt = (
            'You are an expert document analyst. Return strict JSON only with keys: '\
            'summary, bullet_points, entities, tags, recommendations, risk_level, confidence. '\
            'risk_level must be one of low, moderate, high. confidence must be a number between 0 and 1.'
        )
        user = {
            'title': title,
            'metadata': metadata,
            'extracted': extracted,
            'text': text[:12000],
        }
        result = self.chat(
            [
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.1,
            max_tokens=900,
            response_format={'type': 'json_object'},
        )
        try:
            return json.loads(result['content'])
        except json.JSONDecodeError as exc:
            raise FeatherlessError(f'Failed to parse Featherless JSON output: {result["content"][:500]}') from exc

    def answer_with_context(self, *, question: str, context_chunks: List[Dict[str, Any]], document_title: Optional[str] = None) -> Dict[str, Any]:
        context_text = '\n\n'.join(
            f"[Chunk {i+1} | score={chunk.get('score', 0):.3f} | source={chunk.get('document_title')}]\n{chunk.get('text', '')[:1500]}"
            for i, chunk in enumerate(context_chunks[:8])
        )
        system = (
            'You are a retrieval-augmented assistant. Answer only from the supplied context and keep the answer grounded in the uploaded PDF content. '\
            'If the context is insufficient, say that clearly instead of guessing. '\
            'Return strict JSON with keys: answer, citations, follow_up_questions. '\
            'Citations must be short grounded references such as chunk numbers plus a brief snippet. '\
            'Follow_up_questions must be dynamic and specific to the provided document, not generic.'
        )
        user = {
            'document_title': document_title,
            'question': question,
            'context': context_text,
        }
        result = self.chat(
            [
                {'role': 'system', 'content': system},
                {'role': 'user', 'content': json.dumps(user, ensure_ascii=False)},
            ],
            temperature=0.2,
            max_tokens=1000,
            response_format={'type': 'json_object'},
        )
        try:
            return json.loads(result['content'])
        except json.JSONDecodeError as exc:
            raise FeatherlessError(f'Failed to parse Featherless JSON answer: {result["content"][:500]}') from exc
