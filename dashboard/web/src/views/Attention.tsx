import { useEffect, useState } from 'react'
import { api, type AttentionResult, type RunSummary } from '../api'
import Heatmap from '../components/Heatmap'

export default function Attention({ runs }: { runs: RunSummary[] }) {
  const [run, setRun] = useState<string>(runs[0]?.name ?? '')
  const [text, setText] = useState('그는 아버지의 집으로 돌아가서 어머니를 만났다.')
  const [result, setResult] = useState<AttentionResult | null>(null)
  const [focus, setFocus] = useState<[number, number] | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!run && runs[0]) setRun(runs[0].name)
  }, [runs])

  const go = async () => {
    if (!run || !text.trim()) return
    setBusy(true); setError(null)
    try { setResult(await api.attention(run, text)); setFocus(null) } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  useEffect(() => { if (run) void go() }, [run])

  return (
    <div>
      <div className="flex items-baseline justify-between mb-4">
        <h1 className="h1">어텐션</h1>
        <span className="label">행 = 보는 자리 · 열 = 보이는 자리 · 상삼각 0 = 인과 마스크</span>
      </div>
      <div className="card mb-4">
        <div className="flex gap-3 items-center">
          <select className="field" style={{ width: 200 }} value={run} onChange={(e) => setRun(e.target.value)}>
            {runs.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}
          </select>
          <input className="field" value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && go()} placeholder="문장을 넣으세요" />
          <button className="btn-primary shrink-0" onClick={go} disabled={busy || !run}>{busy ? '계산 중…' : '보기'}</button>
        </div>
        {error && <div className="mt-2" style={{ color: 'var(--danger)' }}>{error}</div>}
        {result && (
          <div className="mt-3">
            {result.tokens.map((t, i) => <span key={i} className="chip prompt">{t.replace(/ /g, '␣')}</span>)}
            <span className="label ml-2">{result.tokens.length} 토큰 · {result.n_layer}층 × {result.n_head}헤드</span>
          </div>
        )}
      </div>
      {result && (
        <div className="flex flex-col gap-4">
          {focus && (
            <div className="card">
              <div className="flex items-baseline justify-between">
                <h2 className="card-title">층 {focus[0]} · 헤드 {focus[1]}</h2>
                <button className="btn-ghost" onClick={() => setFocus(null)}>닫기</button>
              </div>
              <div style={{ overflowX: 'auto' }}>
                <Heatmap matrix={result.maps[focus[0]][focus[1]]} labels={result.tokens} cell={Math.max(10, Math.min(22, Math.floor(560 / result.tokens.length)))} />
              </div>
              <div className="label mt-2">셀에 마우스를 올리면 가중치가 보입니다. 밝을수록 그 자리를 많이 본다.</div>
            </div>
          )}
          <div className="card" style={{ overflowX: 'auto' }}>
            <h2 className="card-title">층 × 헤드 (클릭하면 확대)</h2>
            <div className="grid gap-3" style={{ gridTemplateColumns: `repeat(${result.n_head}, max-content)` }}>
              {result.maps.map((layer, li) =>
                layer.map((m, hi) => (
                  <div key={`${li}-${hi}`} onClick={() => setFocus([li, hi])} style={{ cursor: 'pointer', outline: focus && focus[0] === li && focus[1] === hi ? '1px solid var(--accent)' : 'none', padding: 4, borderRadius: 6 }}>
                    <div className="label mb-1">L{li} · H{hi}</div>
                    <Heatmap matrix={m} labels={result.tokens} cell={Math.max(3, Math.min(8, Math.floor(160 / result.tokens.length)))} showLabels={false} />
                  </div>
                )),
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
