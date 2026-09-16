import { useEffect, useState } from 'react'
import Runs from './views/Runs'
import Attention from './views/Attention'
import Generate from './views/Generate'
import Book from './views/Book'
import Notebooks from './views/Notebooks'
import { api, type RunSummary } from './api'

type View = 'runs' | 'attention' | 'generate' | 'book' | 'notebooks'
const NAV: { id: View; label: string; hint: string }[] = [
  { id: 'runs', label: '실험', hint: '손실 곡선' },
  { id: 'attention', label: '어텐션', hint: '층·헤드별 시선' },
  { id: 'generate', label: '생성', hint: '토큰별 확률' },
  { id: 'book', label: '교재', hint: '0~10장' },
  { id: 'notebooks', label: '노트북', hint: '실행 결과' },
]

function viewFromHash(): View {
  const h = location.hash.replace('#/', '') as View
  return NAV.some((n) => n.id === h) ? h : 'runs'
}

export default function App() {
  const [view, setView] = useState<View>(viewFromHash)
  const [runs, setRuns] = useState<RunSummary[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const onHash = () => setView(viewFromHash())
    window.addEventListener('hashchange', onHash)
    return () => window.removeEventListener('hashchange', onHash)
  }, [])
  useEffect(() => {
    api.runs().then(setRuns).catch((e) => setError(String(e)))
  }, [])

  const withCkpt = runs.filter((r) => r.has_checkpoint)
  return (
    <div className="flex h-full app">
      <aside className="sidebar w-[220px] shrink-0 border-r px-3 py-4 flex flex-col gap-1" style={{ borderColor: 'var(--hairline)' }}>
        <div className="px-3 mb-3 brand">
          <div style={{ fontSize: 15, fontWeight: 590, letterSpacing: -0.1 }}>sh-llm-study</div>
          <div className="label">소형 한글 GPT 대시보드</div>
        </div>
        {NAV.map((n) => (
          <a key={n.id} href={`#/${n.id}`} className={`nav-item ${view === n.id ? 'selected' : ''}`}>
            <span>{n.label}</span>
            <span className="label ml-auto">{n.hint}</span>
          </a>
        ))}
        <div className="mt-auto px-3 label footer">
          run {runs.length}개 · 체크포인트 {withCkpt.length}개
        </div>
      </aside>
      <main className="flex-1 overflow-auto">
        <div className="mx-auto max-w-[1280px] px-6 py-5 content">
          {error && (
            <div className="card mb-4" style={{ borderColor: 'var(--danger)' }}>
              API 에 연결할 수 없습니다: <span className="mono">{error}</span>
            </div>
          )}
          {view === 'runs' && <Runs runs={runs} />}
          {view === 'attention' && <Attention runs={withCkpt} />}
          {view === 'generate' && <Generate runs={withCkpt} />}
          {view === 'book' && <Book />}
          {view === 'notebooks' && <Notebooks />}
        </div>
      </main>
    </div>
  )
}
