import { useEffect, useState } from 'react'

interface Chapter { name: string; title: string }
interface Rendered { name: string; title: string; html: string; css: string }

export default function Book() {
  const [chapters, setChapters] = useState<Chapter[]>([])
  const [current, setCurrent] = useState<string | null>(null)
  const [doc, setDoc] = useState<Rendered | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetch('/api/book').then((r) => r.json()).then((c: Chapter[]) => { setChapters(c); if (!current && c.length) setCurrent(c[0].name) }).catch((e) => setError(String(e)))
  }, [])
  useEffect(() => {
    if (!current) return
    fetch(`/api/book/${current}`).then((r) => r.json()).then(setDoc).catch((e) => setError(String(e)))
    window.scrollTo(0, 0)
  }, [current])

  return (
    <div>
      <div className="flex items-baseline justify-between mb-4">
        <h1 className="h1">교재</h1>
        <span className="label">docs/book/*.md · PDF 는 바탕화면 sh-llm-study-book-v*.pdf</span>
      </div>
      {error && <div className="card mb-4" style={{ borderColor: 'var(--danger)' }}>{error}</div>}
      <div className="grid gap-4 two-col" style={{ gridTemplateColumns: '260px 1fr' }}>
        <div className="card" style={{ padding: 8, alignSelf: 'start', position: 'sticky', top: 20 }}>
          {chapters.map((c) => (
            <a key={c.name} href="#/book" onClick={(e) => { e.preventDefault(); setCurrent(c.name) }} className={`nav-item ${c.name === current ? 'selected' : ''}`} style={{ height: 'auto', padding: '6px 12px', lineHeight: 1.35 }}>
              <span style={{ fontSize: 12.5 }}>{c.title}</span>
            </a>
          ))}
        </div>
        <div className="card prose-dark" style={{ padding: '24px 32px', minHeight: 400 }}>
          {doc ? (
            <>
              <style>{doc.css}</style>
              <div dangerouslySetInnerHTML={{ __html: doc.html }} />
            </>
          ) : (
            <div className="label">불러오는 중…</div>
          )}
        </div>
      </div>
    </div>
  )
}
