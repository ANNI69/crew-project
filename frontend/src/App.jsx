import { useEffect, useRef, useState } from 'react'

const API = '/api'

export default function App() {
  const [docs, setDocs] = useState([])
  const [question, setQuestion] = useState('')
  const [result, setResult] = useState(null)
  const [status, setStatus] = useState(null)
  const [busy, setBusy] = useState(false)
  const fileRef = useRef()

  async function loadDocs() {
    const res = await fetch(`${API}/documents`)
    setDocs(await res.json())
  }

  useEffect(() => {
    loadDocs()
  }, [])

  async function upload(e) {
    const file = e.target.files?.[0]
    if (!file) return
    setBusy(true)
    setStatus(null)
    const form = new FormData()
    form.append('file', file)
    const res = await fetch(`${API}/documents`, { method: 'POST', body: form })
    const body = await res.json()
    setBusy(false)
    if (res.ok) {
      setStatus(`Ingested "${body.name}" (${body.chunks} chunks).`)
      fileRef.current.value = ''
      loadDocs()
    } else {
      setStatus(`Ingest failed: ${body.detail}`)
    }
  }

  async function remove(docId) {
    await fetch(`${API}/documents/${docId}`, { method: 'DELETE' })
    loadDocs()
  }

  async function ask(e) {
    e.preventDefault()
    if (!question.trim()) return
    setBusy(true)
    setResult(null)
    const res = await fetch(`${API}/ask`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    })
    const body = await res.json()
    setBusy(false)
    if (res.ok) {
      setResult({ ...body, question })
    } else {
      setStatus(`Ask failed: ${body.detail}`)
    }
  }

  return (
    <main className="container">
      <h1>Document Q&amp;A</h1>
      <p className="subtitle">
        Upload PDF, TXT, or Markdown documents, then ask questions grounded in their content.
      </p>

      <section className="card">
        <h2>Documents</h2>
        <input ref={fileRef} type="file" accept=".pdf,.txt,.md,.markdown" onChange={upload} />
        {docs.length === 0 ? (
          <p className="muted">No documents ingested yet.</p>
        ) : (
          <ul className="doc-list">
            {docs.map((doc) => (
              <li key={doc.doc_id}>
                <span>
                  {doc.name} <span className="muted">({doc.chunks} chunks)</span>
                </span>
                <button className="link" onClick={() => remove(doc.doc_id)}>
                  remove
                </button>
              </li>
            ))}
          </ul>
        )}
        {status && <p className="status">{status}</p>}
      </section>

      <section className="card">
        <h2>Ask</h2>
        <form onSubmit={ask} className="ask-form">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="e.g. What is the refund policy?"
          />
          <button type="submit" disabled={busy}>
            {busy ? 'Working…' : 'Ask'}
          </button>
        </form>

        {result && (
          <div className="answer">
            {result.grounded ? (
              <>
                <h3>Answer</h3>
                <p>{result.answer}</p>
                <h3>Sources</h3>
                <ul className="source-list">
                  {result.sources.map((src, i) => (
                    <li key={i}>
                      <strong>{src.document}</strong>
                      <span className="muted"> · score {src.score}</span>
                      <blockquote>{src.excerpt}</blockquote>
                    </li>
                  ))}
                </ul>
              </>
            ) : (
              <p className="unavailable">⚠ {result.answer}</p>
            )}
          </div>
        )}
      </section>
    </main>
  )
}
