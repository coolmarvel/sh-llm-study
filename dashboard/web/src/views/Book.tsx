import { useEffect, useMemo, useRef, useState } from 'react'
import SourcePanel from '../components/SourcePanel'

interface Chapter { name: string; title: string }
interface Rendered { name: string; title: string; html: string; css: string }
interface Heading { id: string; text: string; level: number }
const READ_KEY = 'shllm.book.read'
const isSourcePath = (t: string) => /^(src|scripts|configs|tests|dashboard)\/[\w./-]+\.(py|yaml|ts|tsx|sh)$/.test(t)

export default function Book() {
  const [chapters, setChapters] = useState<Chapter[]>([])
  const [current, setCurrent] = useState<string | null>(() => location.hash.split('/')[2] || null)
  const [doc, setDoc] = useState<Rendered | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [source, setSource] = useState<string | null>(null)
  const [read, setRead] = useState<Set<string>>(() => { try { return new Set(JSON.parse(localStorage.getItem(READ_KEY) ?? '[]')) } catch { return new Set() } })
  const [active, setActive] = useState<string | null>(null)
  const bodyRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    fetch('/api/book').then((r) => r.json()).then((c: Chapter[]) => { setChapters(c); if (!current && c.length) setCurrent(c[0].name) }).catch((e) => setError(String(e)))
  }, [])
  useEffect(() => {
    if (!current) return
    location.hash = `#/book/${current}`
    setSource(null)
    fetch(`/api/book/${current}`).then((r) => r.json()).then(setDoc).catch((e) => setError(String(e)))
    window.scrollTo(0, 0)
  }, [current])

  // 본문 안의 h2/h3 로 절 목차, 코드 경로 클릭 → 소스 패널
  const headings: Heading[] = useMemo(() => {
    if (!doc) return []
    const div = document.createElement('div'); div.innerHTML = doc.html
    return [...div.querySelectorAll('h2, h3')].map((h) => ({ id: h.id, text: h.textContent ?? '', level: h.tagName === 'H2' ? 2 : 3 }))
  }, [doc])
  useEffect(() => {
    const el = bodyRef.current
    if (!el) return
    const onClick = (e: MouseEvent) => {
      const t = (e.target as HTMLElement).closest('code')
      if (t && isSourcePath(t.textContent ?? '')) { e.preventDefault(); setSource(t.textContent!) }
    }
    el.addEventListener('click', onClick)
    el.querySelectorAll('code').forEach((c) => { if (isSourcePath(c.textContent ?? '')) c.classList.add('src-link') })
    const obs = new IntersectionObserver((entries) => { for (const en of entries) if (en.isIntersecting) setActive((en.target as HTMLElement).id) }, { rootMargin: '0px 0px -70% 0px' })
    el.querySelectorAll('h2, h3').forEach((h) => obs.observe(h))
    return () => { el.removeEventListener('click', onClick); obs.disconnect() }
  }, [doc])

  const idx = chapters.findIndex((c) => c.name === current)
  const markRead = () => { const n = new Set(read); n.add(current!); setRead(n); localStorage.setItem(READ_KEY, JSON.stringify([...n])) }
  const goto = (i: number) => { if (i >= 0 && i < chapters.length) setCurrent(chapters[i].name) }

  return (
    <div>
      <div className="flex items-baseline justify-between mb-4">
        <h1 className="h1">교재</h1>
        <span className="label">읽음 {read.size}/{chapters.length} · 코드 경로를 누르면 소스가 열립니다 · PDF 는 바탕화면</span>
      </div>
      {error && <div className="card mb-4" style={{ borderColor: 'var(--danger)' }}>{error}</div>}
      <div className="book-layout">
        <aside className="card book-chapters">
          {chapters.map((c, i) => (
            <a key={c.name} href={`#/book/${c.name}`} onClick={(e) => { e.preventDefault(); setCurrent(c.name) }} className={`nav-item ${c.name === current ? 'selected' : ''}`} style={{ height: 'auto', padding: '6px 10px', lineHeight: 1.35 }}>
              <span className="mono label" style={{ width: 22, flexShrink: 0 }}>{String(i).padStart(2, '0')}</span>
              <span style={{ fontSize: 12.5 }}>{c.title.replace(/^\d+장 /, '')}</span>
              {read.has(c.name) && <span className="ml-auto" style={{ color: 'var(--accent)' }}>✓</span>}
            </a>
          ))}
        </aside>
        <div className="book-main">
          <article className="card prose-dark book-article" ref={bodyRef}>
            {doc ? (<><style>{doc.css}</style><div dangerouslySetInnerHTML={{ __html: doc.html }} /></>) : <div className="label">불러오는 중…</div>}
            {doc && (
              <div className="flex items-center gap-2 mt-8 pt-4" style={{ borderTop: '1px solid var(--hairline)' }}>
                <button className="btn-ghost" onClick={() => goto(idx - 1)} disabled={idx <= 0}>← 이전 장</button>
                <button className="btn-primary" onClick={markRead}>{read.has(current!) ? '읽음 ✓' : '이 장 읽음으로 표시'}</button>
                <button className="btn-ghost ml-auto" onClick={() => goto(idx + 1)} disabled={idx >= chapters.length - 1}>다음 장 →</button>
              </div>
            )}
          </article>
          {source && <SourcePanel path={source} onClose={() => setSource(null)} />}
        </div>
        <aside className="book-toc">
          <div className="label mb-2">이 장의 절</div>
          {headings.map((h) => (
            <a key={h.id} href={`#${h.id}`} onClick={(e) => { e.preventDefault(); document.getElementById(h.id)?.scrollIntoView({ block: 'start' }) }} className={`toc-item ${h.level === 3 ? 'sub' : ''} ${active === h.id ? 'active' : ''}`}>{h.text}</a>
          ))}
        </aside>
      </div>
    </div>
  )
}
