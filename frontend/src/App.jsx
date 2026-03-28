import React, { useEffect, useMemo, useState } from 'react'

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
const PAGES = [
  { key: 'ingest', label: 'Ingest & Document View' },
  { key: 'semantic', label: 'Semantic Search' },
  { key: 'grounded', label: 'Ask Grounded Questions' },
  { key: 'agents', label: 'Agent Workflow' },
]

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, options)
  if (!res.ok) {
    const text = await res.text()
    throw new Error(text || `Request failed: ${res.status}`)
  }
  return res.json()
}

function pct(score) {
  if (score == null || Number.isNaN(Number(score))) return '—'
  return `${(Number(score) * 100).toFixed(1)}%`
}

function labDefault(activeDocument, name, fallback = '') {
  return activeDocument?.extracted?.labs?.[name]?.value ?? fallback
}

function JsonBlock({ data }) {
  return <pre>{JSON.stringify(data, null, 2)}</pre>
}

export default function App() {
  const [page, setPage] = useState('ingest')
  const [health, setHealth] = useState(null)
  const [documents, setDocuments] = useState([])
  const [activeDocument, setActiveDocument] = useState(null)
  const [title, setTitle] = useState('')
  const [text, setText] = useState('')
  const [file, setFile] = useState(null)
  const [searchQuery, setSearchQuery] = useState('diagnosis')
  const [searchResults, setSearchResults] = useState([])
  const [suggestedQuestions, setSuggestedQuestions] = useState([])
  const [groundedSearchQuery, setGroundedSearchQuery] = useState('')
  const [question, setQuestion] = useState('What is the most likely diagnosis and why?')
  const [answer, setAnswer] = useState(null)
  const [agentResult, setAgentResult] = useState(null)
  const [dragActive, setDragActive] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [simulation, setSimulation] = useState({ hemoglobin: '', wbc: '', platelets: '' })

  const hasInput = useMemo(() => Boolean(text.trim() || file), [text, file])
  const configured = Boolean(health?.featherless?.configured)

  useEffect(() => {
    refreshAll()
  }, [])

  async function refreshAll() {
    try {
      const [healthData, docsData] = await Promise.all([request('/api/health'), request('/api/documents')])
      setHealth(healthData)
      setDocuments(docsData.items || [])
      if (!activeDocument && (docsData.items || []).length > 0) {
        await openDocument(docsData.items[0].document_id)
      }
    } catch (err) {
      setError(err.message)
    }
  }

  async function openDocument(documentId) {
    setLoading(true)
    setError('')
    try {
      const data = await request(`/api/documents/${documentId}`)
      setActiveDocument(data)
      setSimulation({
        hemoglobin: labDefault(data, 'hemoglobin', ''),
        wbc: labDefault(data, 'wbc', ''),
        platelets: labDefault(data, 'platelets', ''),
      })
      setAnswer(null)
      setAgentResult(null)
      setSearchResults([])
      setSuggestedQuestions(data.suggested_questions || [])
      setGroundedSearchQuery('')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function ingest() {
    if (!hasInput) return
    setLoading(true)
    setError('')
    try {
      let data
      if (file) {
        const body = new FormData()
        body.append('file', file)
        if (title.trim()) body.append('title', title.trim())
        data = await request('/api/ingest/file', { method: 'POST', body })
      } else {
        data = await request('/api/ingest/text', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ title: title.trim(), text }),
        })
      }
      setText('')
      setTitle('')
      setFile(null)
      await refreshAll()
      await openDocument(data.document_id)
      setPage('semantic')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function runSearch() {
    if (!searchQuery.trim()) return
    setLoading(true)
    setError('')
    try {
      const data = await request('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: searchQuery, document_id: activeDocument?.document_id || null, top_k: 8 }),
      })
      setSearchResults(data.items || [])
      setSuggestedQuestions(data.suggested_questions || [])
      setGroundedSearchQuery(data.grounded_query || '')
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function askQuestion() {
    if (!question.trim()) return
    setLoading(true)
    setError('')
    try {
      const data = await request('/api/ask', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, document_id: activeDocument?.document_id || null, top_k: 6 }),
      })
      setAnswer(data)
      setSuggestedQuestions(data.follow_up_questions || suggestedQuestions)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  async function runAgents() {
    if (!question.trim()) return
    setLoading(true)
    setError('')
    try {
      const overrides = Object.fromEntries(
        Object.entries(simulation)
          .filter(([, value]) => `${value}`.trim() !== '')
          .map(([key, value]) => [key, Number(value)])
      )
      const data = await request('/api/agents/sync', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, document_id: activeDocument?.document_id || null, top_k: 6, simulation_overrides: overrides }),
      })
      setAgentResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  function onDrop(event) {
    event.preventDefault()
    setDragActive(false)
    const dropped = event.dataTransfer.files?.[0]
    if (dropped) setFile(dropped)
  }

  const apiExamples = useMemo(() => {
    const payload = {
      question,
      document_id: activeDocument?.document_id || null,
      top_k: 6,
      simulation_overrides: Object.fromEntries(
        Object.entries(simulation)
          .filter(([, value]) => `${value}`.trim() !== '')
          .map(([key, value]) => [key, Number(value)])
      ),
    }
    return {
      sync: `curl -X POST ${API_BASE}/api/agents/sync -H 'Content-Type: application/json' -d '${JSON.stringify(payload)}'`,
      ask: `curl -X POST ${API_BASE}/api/ask -H 'Content-Type: application/json' -d '${JSON.stringify({ question, document_id: activeDocument?.document_id || null, top_k: 6 })}'`,
    }
  }, [question, activeDocument, simulation])

  function renderIngestPage() {
    return (
      <section className="page-grid two-col">
        <div className="card">
          <div className="section-head">
            <div>
              <div className="section-title">1. Ingest text or file</div>
              <div className="small muted">PDF, TXT, DOCX, JSON, HTML, XML, or pasted text</div>
            </div>
            <button onClick={ingest} disabled={loading || !hasInput}>{loading ? 'Working...' : 'Ingest'}</button>
          </div>

          <div className={`dropzone ${dragActive ? 'drag' : ''}`} onDragOver={(e) => { e.preventDefault(); setDragActive(true) }} onDragLeave={() => setDragActive(false)} onDrop={onDrop}>
            <input type="file" accept=".pdf,.txt,.docx,.json,.html,.htm,.xml" onChange={(e) => setFile(e.target.files?.[0] || null)} />
            <div className="small muted">Drag and drop a file here or use the picker.</div>
            {file && <div className="selected-file">Selected: {file.name}</div>}
          </div>

          <div className="form-grid">
            <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="Optional title" />
            <textarea rows="8" value={text} onChange={(e) => setText(e.target.value)} placeholder="Or paste text content here..." />
          </div>
        </div>

        <div className="card tall">
          <div className="section-head">
            <div>
              <div className="section-title">Document intelligence view</div>
              <div className="small muted">AI summary, extracted entities, chunk preview, and raw preview</div>
            </div>
          </div>

          {!activeDocument ? <div className="muted small">Select or ingest a document to inspect it.</div> : (
            <div className="stack gap-md">
              <div>
                <div className="meta-row">
                  <h2>{activeDocument.title}</h2>
                  <span className={`pill ${activeDocument.ai_summary?.risk_level || 'low'}`}>{activeDocument.ai_summary?.risk_level || 'low'}</span>
                </div>
                <div className="small muted">{activeDocument.source_type} {activeDocument.source_name ? `· ${activeDocument.source_name}` : ''}</div>
              </div>

              <div>
                <div className="section-title mini">AI summary</div>
                <p>{activeDocument.ai_summary?.summary}</p>
                <ul>
                  {(activeDocument.ai_summary?.bullet_points || []).map((item, idx) => <li key={idx}>{item}</li>)}
                </ul>
              </div>

              <div>
                <div className="section-title mini">Tags</div>
                <div className="tag-wrap">
                  {(activeDocument.ai_summary?.tags || []).map((item) => <span className="tag" key={item}>{item}</span>)}
                </div>
              </div>

              <div>
                <div className="section-title mini">Extracted labs</div>
                <div className="evidence-grid">
                  {Object.entries(activeDocument.extracted?.labs || {}).map(([name, item]) => (
                    <div className="field-card" key={name}>
                      <strong>{name}</strong>
                      <span className={`pill ${item.status === 'normal' ? 'low' : 'high'}`}>{item.status}</span>
                      <div>{item.value} {item.unit}</div>
                      <div className="small muted">range {item.reference_range?.min}–{item.reference_range?.max}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <div className="section-title mini">Entity extraction</div>
                {Object.entries(activeDocument.extracted?.entities || {}).map(([group, values]) => (
                  <div key={group} className="entity-group">
                    <strong>{group}</strong>
                    <div className="tag-wrap">
                      {values.length ? values.map((value) => <span className="tag secondary" key={value}>{value}</span>) : <span className="small muted">none</span>}
                    </div>
                  </div>
                ))}
              </div>

              <div>
                <div className="section-title mini">Vector chunks</div>
                <div className="chunk-list">
                  {(activeDocument.chunks || []).slice(0, 6).map((chunk) => (
                    <details key={chunk.chunk_id} className="chunk-item">
                      <summary>Chunk #{chunk.chunk_index + 1} · ~{chunk.token_estimate} tokens</summary>
                      <p>{chunk.text}</p>
                    </details>
                  ))}
                </div>
              </div>

              <div>
                <div className="section-title mini">Raw preview</div>
                <pre>{activeDocument.preview_text}</pre>
              </div>
            </div>
          )}
        </div>
      </section>
    )
  }

  function renderSemanticPage() {
    return (
      <section className="page-grid single-col">
        <div className="card">
          <div className="section-head">
            <div>
              <div className="section-title">2. Semantic search</div>
              <div className="small muted">Run retrieval over the active PDF with document-aware query grounding</div>
            </div>
            <button onClick={runSearch} disabled={loading || !searchQuery.trim()}>Search</button>
          </div>
          <div className="inline-form">
            <input value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} placeholder="Search diagnosis, symptom clusters, medications, sections..." />
          </div>
          {!!groundedSearchQuery && <div className="result-card top-gap"><strong>Grounded query</strong><p className="small muted top-gap">{groundedSearchQuery}</p></div>}

          {!!suggestedQuestions.length && (
            <div className="top-gap">
              <div className="section-title mini">Suggested grounded prompts</div>
              <div className="chip-row">
                {suggestedQuestions.map((item) => (
                  <button key={item} className="chip" onClick={() => { setQuestion(item); setPage('grounded') }}>{item}</button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="card tall">
          <div className="section-head">
            <div>
              <div className="section-title">Retrieved chunks</div>
              <div className="small muted">Best matching evidence from the active PDF dataset</div>
            </div>
          </div>
          <div className="results-list min-panel">
            {searchResults.length === 0 ? <div className="muted small">No search results yet.</div> : searchResults.map((item) => (
              <div key={item.chunk_id} className="result-card">
                <div className="meta-row"><strong>{item.document_title}</strong><span className="small muted">match {pct(item.score)}</span></div>
                <p>{item.text}</p>
              </div>
            ))}
          </div>
        </div>
      </section>
    )
  }

  function renderGroundedPage() {
    return (
      <section className="page-grid two-col">
        <div className="card">
          <div className="section-head">
            <div>
              <div className="section-title">3. Ask grounded questions</div>
              <div className="small muted">Ask questions that stay anchored to the retrieved PDF evidence</div>
            </div>
            <button className="secondary-btn" onClick={askQuestion} disabled={loading || !question.trim()}>Ask</button>
          </div>

          <textarea rows="5" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="What is the most likely diagnosis and why?" />
          {!!answer?.grounded_query && <div className="small muted top-gap">Answer grounded with: {answer.grounded_query}</div>}

          {!!suggestedQuestions.length && (
            <div className="top-gap">
              <div className="section-title mini">Try one of these</div>
              <div className="chip-row">
                {suggestedQuestions.map((item) => (
                  <button key={item} className="chip" onClick={() => setQuestion(item)}>{item}</button>
                ))}
              </div>
            </div>
          )}
        </div>

        <div className="card tall">
          <div className="section-head">
            <div>
              <div className="section-title">Grounded answer</div>
              <div className="small muted">Answer, citations, and follow-up questions from the active PDF</div>
            </div>
          </div>

          <div className="answer-box top-gap">
            {!answer ? <div className="muted small">No answer yet.</div> : (
              <div className="stack gap-md">
                <p>{answer.answer}</p>
                {!!answer.citations?.length && <div><strong>Citations:</strong> {answer.citations.join(', ')}</div>}
                {!!answer.follow_up_questions?.length && (
                  <div>
                    <strong>Follow-ups</strong>
                    <ul>
                      {answer.follow_up_questions.map((item) => <li key={item}>{item}</li>)}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </section>
    )
  }

  function renderAgentsPage() {
    return (
      <section className="page-grid two-col">
        <div className="card">
          <div className="section-head">
            <div>
              <div className="section-title">4. Agent workflow</div>
              <div className="small muted">Show how specialized agents reason over the same grounded evidence</div>
            </div>
            <button onClick={runAgents} disabled={loading || !question.trim()}>Run Agent Sync</button>
          </div>

          <div className="top-gap">
            <div className="section-title mini">Shared question</div>
            <textarea rows="4" value={question} onChange={(e) => setQuestion(e.target.value)} placeholder="What is the most likely diagnosis and why?" />
          </div>

          <div className="section-title mini top-gap">Simulation mode</div>
          <div className="sim-grid">
            {['hemoglobin', 'wbc', 'platelets'].map((key) => (
              <label key={key} className="field-card">
                <span>{key}</span>
                <input type="number" step="0.1" value={simulation[key]} onChange={(e) => setSimulation((prev) => ({ ...prev, [key]: e.target.value }))} />
              </label>
            ))}
          </div>

          <div className="top-gap stack gap-md">
            <div>
              <div className="section-title mini">POST /api/agents/sync</div>
              <pre>{apiExamples.sync}</pre>
            </div>
            <div>
              <div className="section-title mini">POST /api/ask</div>
              <pre>{apiExamples.ask}</pre>
            </div>
          </div>
        </div>

        <div className="card tall">
          <div className="section-head">
            <div>
              <div className="section-title">Live workflow output</div>
              <div className="small muted">Timeline, evidence cards, and final summary</div>
            </div>
            <span className="pill highlite">Live sync</span>
          </div>

          {!agentResult ? <div className="muted small">Run Agent Sync to visualize ingestion, retrieval, diagnosis, validation, critique, and explanation.</div> : (
            <div className="stack gap-md">
              <div className="result-card">
                <div className="meta-row"><strong>Final output</strong><span className="small muted">confidence {pct(agentResult.final?.confidence)}</span></div>
                <p>{agentResult.final?.summary}</p>
                <div className="chip-row">
                  {(agentResult.final?.top_diagnoses || []).map((item) => <span className="tag" key={item.name}>{item.name} · {pct(item.confidence)}</span>)}
                </div>
              </div>

              <div className="timeline">
                {(agentResult.workflow || []).map((step) => (
                  <div className="timeline-item" key={step.agent}>
                    <div className="timeline-dot" />
                    <div className="timeline-content">
                      <div className="meta-row"><strong>{step.title}</strong><span className="small muted">{step.status}</span></div>
                      <p className="small">{step.summary}</p>
                      <JsonBlock data={step.details} />
                    </div>
                  </div>
                ))}
              </div>

              <div>
                <div className="section-title mini">Evidence cards</div>
                <div className="evidence-grid">
                  {(agentResult.evidence || []).map((item, idx) => (
                    <div className="field-card" key={`${item.label}-${idx}`}>
                      <strong>{item.label}</strong>
                      <span className={`pill ${item.status === 'low' ? 'high' : item.status === 'normal' ? 'low' : 'moderate'}`}>{String(item.status)}</span>
                      <div className="small muted">{String(item.value)}</div>
                      <div className="small">{item.reason}</div>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <div className="section-title mini">Response preview</div>
                <JsonBlock data={agentResult?.final || { hint: 'Run Agent Sync to populate a live response example.' }} />
              </div>
            </div>
          )}
        </div>
      </section>
    )
  }

  function renderPage() {
    if (page === 'semantic') return renderSemanticPage()
    if (page === 'grounded') return renderGroundedPage()
    if (page === 'agents') return renderAgentsPage()
    return renderIngestPage()
  }

  return (
    <div className="layout">
      <aside className="sidebar card">
        <div>
          <div className="badge">Hackathon-ready Multi-Agent RAG</div>
          <h1>Clinical Intelligence Copilot</h1>
          <p className="muted">Now split into dedicated pages for ingest, semantic retrieval, grounded Q&A, and agent workflow demos.</p>
        </div>

        <div className="stack gap-sm">
          <div className="meta-row"><span>Status</span><strong>{health?.status || 'loading...'}</strong></div>
          <div className="meta-row"><span>Mode</span><strong>{health?.featherless?.mode || 'loading...'}</strong></div>
          <div className="meta-row"><span>LLM</span><strong>{configured ? 'live' : 'fallback'}</strong></div>
          <div className="meta-row"><span>Model</span><strong className="ellipsis">{health?.featherless?.model || '—'}</strong></div>
          <div className="meta-row"><span>Docs</span><strong>{documents.length}</strong></div>
        </div>

        <div className="stack gap-sm">
          <div className="section-title">Pages</div>
          <div className="nav-list">
            {PAGES.map((item) => (
              <button key={item.key} className={`nav-button ${page === item.key ? 'active' : ''}`} onClick={() => setPage(item.key)}>
                {item.label}
              </button>
            ))}
          </div>
        </div>

        <div className="stack gap-sm">
          <div className="section-title">Indexed documents</div>
          <div className="doc-list">
            {documents.length === 0 && <div className="muted small">No documents indexed yet.</div>}
            {documents.map((doc) => (
              <button key={doc.document_id} className={`doc-button ${activeDocument?.document_id === doc.document_id ? 'active' : ''}`} onClick={() => openDocument(doc.document_id)}>
                <div>
                  <div className="doc-title">{doc.title}</div>
                  <div className="small muted">{doc.source_type} · {doc.chunk_count || 0} chunks</div>
                </div>
                <span className={`pill ${doc.risk_level || 'low'}`}>{doc.risk_level || 'low'}</span>
              </button>
            ))}
          </div>
        </div>
      </aside>

      <main className="main">
        <div className="page-header card">
          <div>
            <div className="section-title">Current page</div>
            <h2>{PAGES.find((item) => item.key === page)?.label}</h2>
            <div className="small muted">{activeDocument ? `Active dataset: ${activeDocument.title}` : 'Load a PDF dataset to begin.'}</div>
          </div>
          {activeDocument && <span className={`pill ${activeDocument.ai_summary?.risk_level || 'low'}`}>{activeDocument.ai_summary?.risk_level || 'low'}</span>}
        </div>

        {renderPage()}

        {error && <div className="error">{error}</div>}
      </main>
    </div>
  )
}
