import { useEffect, useState } from 'react'
import { api, fmt, type GenerateResult, type RunSummary } from '../api'

function Slider({ label, value, min, max, step, onChange, display }: { label: string; value: number; min: number; max: number; step: number; onChange: (v: number) => void; display?: string }) {
  return (
    <label className="block">
      <div className="flex justify-between"><span className="label">{label}</span><span className="mono">{display ?? value}</span></div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={(e) => onChange(Number(e.target.value))} />
    </label>
  )
}

export default function Generate({ runs }: { runs: RunSummary[] }) {
  const [run, setRun] = useState<string>(runs[0]?.name ?? '')
  const [prompt, setPrompt] = useState('옛날 옛적에 호랑이가')
  const [maxNew, setMaxNew] = useState(60)
  const [temperature, setTemperature] = useState(0.8)
  const [topK, setTopK] = useState(50)
  const [topP, setTopP] = useState(1.0)
  const [rep, setRep] = useState(1.1)
  const [seed, setSeed] = useState(0)
  const [result, setResult] = useState<GenerateResult | null>(null)
  const [pick, setPick] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => { if (!run && runs[0]) setRun(runs[0].name) }, [runs])

  const go = async () => {
    if (!run || !prompt.trim()) return
    setBusy(true); setError(null)
    try {
      const r = await api.generate({ run, prompt, max_new_tokens: maxNew, temperature, top_k: topK >= 8192 ? null : topK, top_p: topP >= 1 ? null : topP, repetition_penalty: rep, top_n: 10, seed })
      setResult(r); setPick(r.steps.length ? 0 : null)
    } catch (e) { setError(String(e)) } finally { setBusy(false) }
  }
  const step = result && pick != null ? result.steps[pick] : null
  const show = (s: string) => s.replace(/ /g, '␣').replace(/\n/g, '⏎')
  return (
    <div>
      <div className="flex items-baseline justify-between mb-4">
        <h1 className="h1">생성</h1>
        <span className="label">토큰을 클릭하면 그 자리의 후보 확률이 보입니다</span>
      </div>
      <div className="grid gap-4 two-col" style={{ gridTemplateColumns: '300px 1fr' }}>
        <div className="card flex flex-col gap-3">
          <select className="field" value={run} onChange={(e) => setRun(e.target.value)}>
            {runs.map((r) => <option key={r.name} value={r.name}>{r.name}</option>)}
          </select>
          <textarea className="field" rows={3} value={prompt} onChange={(e) => setPrompt(e.target.value)} />
          <Slider label="temperature" value={temperature} min={0} max={2} step={0.05} onChange={setTemperature} display={temperature.toFixed(2)} />
          <Slider label="top-k (8192 = 끔)" value={topK} min={1} max={8192} step={1} onChange={setTopK} display={topK >= 8192 ? '끔' : String(topK)} />
          <Slider label="top-p (1.00 = 끔)" value={topP} min={0.1} max={1} step={0.01} onChange={setTopP} display={topP >= 1 ? '끔' : topP.toFixed(2)} />
          <Slider label="반복 억제" value={rep} min={1} max={2} step={0.05} onChange={setRep} display={rep.toFixed(2)} />
          <Slider label="길이 (토큰)" value={maxNew} min={10} max={200} step={10} onChange={setMaxNew} />
          <div className="flex gap-2 items-center">
            <span className="label">seed</span>
            <input className="field" style={{ width: 80 }} type="number" value={seed} onChange={(e) => setSeed(Number(e.target.value))} />
            <button className="btn-ghost" onClick={() => setSeed((s) => s + 1)}>+1</button>
            <button className="btn-primary ml-auto" onClick={go} disabled={busy || !run}>{busy ? '생성 중…' : '생성'}</button>
          </div>
          {error && <div style={{ color: 'var(--danger)' }}>{error}</div>}
        </div>
        <div className="flex flex-col gap-4">
          <div className="card" style={{ minHeight: 160 }}>
            <h2 className="card-title">생성문</h2>
            {!result && <div className="label">프롬프트를 넣고 생성을 누르세요.</div>}
            {result && (
              <div style={{ lineHeight: 2 }}>
                {result.prompt_tokens.map((t, i) => <span key={`p${i}`} className="chip prompt">{show(t)}</span>)}
                {result.steps.map((s, i) => (
                  <span key={i} className={`chip ${pick === i ? 'selected' : ''}`} onClick={() => setPick(i)} style={{ opacity: 0.45 + 0.55 * Math.min(1, s.prob * 2) }} title={`p=${fmt.pct(s.prob)}`}>
                    {show(s.token)}
                  </span>
                ))}
              </div>
            )}
          </div>
          {step && (
            <div className="card">
              <div className="flex items-baseline justify-between">
                <h2 className="card-title">자리 {pick! + 1}: <span className="mono">{show(step.token)}</span> 가 뽑힘 (조정 후 {fmt.pct(step.prob)} · 원본 {fmt.pct(step.raw_prob)})</h2>
                <span className="label">상위 10 후보 · 막대 = 조정 후 확률, 흐린 막대 = 원본</span>
              </div>
              <div className="flex flex-col gap-1">
                {step.candidates.map((c, i) => (
                  <div key={i} className="grid items-center gap-2" style={{ gridTemplateColumns: '120px 1fr 60px 60px' }}>
                    <span className="mono truncate" style={{ color: c.token === step.token ? 'var(--fg)' : 'var(--secondary)' }}>{show(c.token)}</span>
                    <div style={{ position: 'relative', height: 14, background: 'var(--canvas)', borderRadius: 3 }}>
                      <div style={{ position: 'absolute', inset: 0, width: `${c.raw_prob * 100}%`, background: 'var(--prob-bar)', opacity: 0.5, borderRadius: 3 }} />
                      <div style={{ position: 'absolute', inset: 0, width: `${c.prob * 100}%`, background: c.token === step.token ? 'var(--accent)' : 'var(--prob-bar)', borderRadius: 3 }} />
                    </div>
                    <span className="mono label text-right">{fmt.pct(c.prob)}</span>
                    <span className="mono label text-right" style={{ color: 'var(--quiet)' }}>{fmt.pct(c.raw_prob)}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
