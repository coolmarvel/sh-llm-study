import { useEffect, useState } from 'react'

interface NB { name: string; title: string; cells: number; executed: boolean; executed_at: number | null; lab_url: string }

export default function Notebooks() {
  const [list, setList] = useState<NB[]>([])
  const [current, setCurrent] = useState<NB | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    fetch('/api/notebooks').then((r) => r.json()).then((l: NB[]) => { setList(l); if (!current && l.length) setCurrent(l[0]) }).catch((e) => setError(String(e)))
  }, [])
  const when = (t: number | null) => (t ? new Date(t * 1000).toLocaleString('ko-KR', { hour12: false }) : '—')
  return (
    <div>
      <div className="flex items-baseline justify-between mb-4">
        <h1 className="h1">노트북</h1>
        <span className="label">실행 결과 = verify.sh 가 만든 build/notebooks · 직접 실행 = JupyterLab(8888)</span>
      </div>
      {error && <div className="card mb-4" style={{ borderColor: 'var(--danger)' }}>{error}</div>}
      <div className="grid gap-4 two-col" style={{ gridTemplateColumns: '300px 1fr' }}>
        <div className="card" style={{ padding: 8, alignSelf: 'start', position: 'sticky', top: 20 }}>
          {list.map((n) => (
            <a key={n.name} href="#/notebooks" onClick={(e) => { e.preventDefault(); setCurrent(n) }} className={`nav-item ${n.name === current?.name ? 'selected' : ''}`} style={{ height: 'auto', padding: '6px 12px', lineHeight: 1.35 }}>
              <span style={{ fontSize: 12.5 }}>{n.title}</span>
              <span className="label ml-auto mono" style={{ fontSize: 11 }}>{n.executed ? '결과 ✓' : '미실행'}</span>
            </a>
          ))}
          <div className="label px-3 pt-3">
            JupyterLab: <a href="http://localhost:8888/lab/tree/notebooks" target="_blank" rel="noreferrer" style={{ color: 'var(--accent)' }}>localhost:8888</a>
          </div>
        </div>
        <div className="flex flex-col gap-3">
          {current && (
            <div className="card flex items-center gap-3" style={{ padding: '10px 16px' }}>
              <span className="mono">{current.name}.ipynb</span>
              <span className="label">{current.cells} 셀 · 실행 결과 {when(current.executed_at)}</span>
              <a className="btn-primary ml-auto flex items-center" href={current.lab_url} target="_blank" rel="noreferrer">JupyterLab 에서 실행</a>
            </div>
          )}
          {current && current.executed ? (
            <iframe title={current.name} src={`/api/notebooks/${current.name}/html`} style={{ width: '100%', height: 'calc(100vh - 180px)', border: '1px solid var(--hairline)', borderRadius: 8, background: '#fff' }} />
          ) : (
            current && (
              <div className="card">
                <p style={{ color: 'var(--secondary)' }}>아직 실행 결과가 없습니다. 오른쪽 위 버튼으로 JupyterLab 에서 직접 실행하거나, <span className="mono">bash scripts/verify.sh</span> 를 돌리면 결과가 여기에 나타납니다.</p>
                <iframe title={current.name} src={`/api/notebooks/${current.name}/html?executed=false`} style={{ width: '100%', height: '60vh', border: '1px solid var(--hairline)', borderRadius: 8, background: '#fff', marginTop: 12 }} />
              </div>
            )
          )}
        </div>
      </div>
    </div>
  )
}
