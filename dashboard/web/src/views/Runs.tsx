import { useEffect, useState } from 'react'
import { CartesianGrid, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { api, fmt, type LogRow, type RunSummary } from '../api'

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="label">{label}</div>
      <div className="stat-value">{value}</div>
    </div>
  )
}

export default function Runs({ runs }: { runs: RunSummary[] }) {
  const [selected, setSelected] = useState<string | null>(null)
  const [compare, setCompare] = useState<Set<string>>(new Set())  // 겹쳐 그릴 다른 run 들
  const [log, setLog] = useState<LogRow[]>([])
  const [others, setOthers] = useState<Record<string, LogRow[]>>({})
  const run = runs.find((r) => r.name === selected) ?? runs[0]

  useEffect(() => {
    if (!run) return
    api.log(run.name).then(setLog).catch(() => setLog([]))
  }, [run?.name])
  useEffect(() => {
    const names = [...compare].filter((n) => n !== run?.name)
    Promise.all(names.map((n) => api.log(n).then((l) => [n, l] as const).catch(() => [n, [] as LogRow[]] as const))).then((pairs) => setOthers(Object.fromEntries(pairs)))
  }, [compare, run?.name])
  const toggleCompare = (name: string) => setCompare((s) => { const n = new Set(s); if (n.has(name)) n.delete(name); else n.add(name); return n })
  // 겹쳐 그리기: step 기준으로 합친 데이터 (각 run 의 val 을 별도 키로)
  const merged: Record<number, Record<string, number>> = {}
  for (const row of log) merged[row.step] = { ...(merged[row.step] ?? {}), step: row.step, train_loss: row.train_loss, val_loss: row.val_loss }
  for (const [n, rows] of Object.entries(others)) for (const row of rows) merged[row.step] = { ...(merged[row.step] ?? {}), step: row.step, [`val_${n}`]: row.val_loss }
  const data = Object.values(merged).sort((a, b) => a.step - b.step)
  const palette = ['#e0a458', '#5dbb8a', '#d46fa1', '#7fc4e8']

  if (!runs.length) {
    return (
      <div className="card">
        <h1 className="h1 mb-2">실험</h1>
        <p style={{ color: 'var(--secondary)' }}>
          아직 run 이 없습니다. <span className="mono">uv run python scripts/train.py --config configs/small-cpu.yaml</span> 로 학습을 시작하면 여기에 나타납니다.
        </p>
      </div>
    )
  }
  const best = log.length ? log.reduce((a, b) => (b.val_loss < a.val_loss ? b : a)) : null
  return (
    <div>
      <div className="flex items-baseline justify-between mb-4">
        <h1 className="h1">실험</h1>
        <span className="label">data/runs/&lt;run&gt;/log.jsonl</span>
      </div>
      <div className="grid gap-4 two-col" style={{ gridTemplateColumns: '420px 1fr' }}>
        <div className="card" style={{ padding: 8 }}>
          <div className="run-row label" style={{ cursor: 'default', height: 28 }}>
            <span>run (☐ 겹쳐 보기)</span><span>파라미터</span><span>step</span><span>best val</span><span>경과</span>
          </div>
          {runs.map((r) => (
            <div key={r.name} className={`run-row ${r.name === run?.name ? 'selected' : ''}`} onClick={() => setSelected(r.name)}>
              <span className="truncate flex items-center gap-1">
                <input type="checkbox" checked={compare.has(r.name)} onChange={() => toggleCompare(r.name)} onClick={(e) => e.stopPropagation()} style={{ accentColor: 'var(--accent)' }} title="이 run 의 val 을 겹쳐 그린다" />
                <span className="dot" />{r.name}</span>
              <span className="mono">{fmt.params(r.n_params)}</span>
              <span className="mono">{r.steps}</span>
              <span className="mono">{fmt.loss(r.best_val)}</span>
              <span className="mono">{fmt.minutes(r.elapsed)}</span>
            </div>
          ))}
        </div>
        <div className="flex flex-col gap-4">
          <div className="card">
            <div className="grid grid-cols-4 gap-4 stats">
              <Stat label="파라미터" value={fmt.params(run.n_params)} />
              <Stat label="step" value={String(run.steps)} />
              <Stat label="best val loss" value={fmt.loss(run.best_val)} />
              <Stat label="경과" value={fmt.minutes(run.elapsed)} />
            </div>
          </div>
          <div className="card">
            <div className="flex items-baseline justify-between">
              <h2 className="card-title">손실 곡선: {run.name}</h2>
              <span className="label mono">
                {String(run.model.n_layer ?? '?')}층 · C={String(run.model.n_embd ?? '?')} · T={String(run.model.block_size ?? '?')}
              </span>
            </div>
            <div style={{ height: 340 }}>
              <ResponsiveContainer>
                <LineChart data={data} margin={{ top: 16, right: 16, bottom: 4, left: 0 }}>
                  <CartesianGrid stroke="var(--hairline)" vertical={false} />
                  <XAxis dataKey="step" stroke="var(--muted)" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} />
                  <YAxis stroke="var(--muted)" tick={{ fontSize: 11, fontFamily: 'var(--font-mono)' }} domain={['auto', 'auto']} width={44} />
                  <Tooltip
                    contentStyle={{ background: 'var(--surface)', border: '1px solid var(--hairline)', borderRadius: 6, fontFamily: 'var(--font-mono)', fontSize: 12 }}
                    labelStyle={{ color: 'var(--muted)' }}
                    formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : String(v))}
                  />
                  <Line type="monotone" dataKey="train_loss" name="train" stroke="var(--series-train)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                  <Line type="monotone" dataKey="val_loss" name="val" stroke="var(--series-val)" strokeWidth={2} dot={false} isAnimationActive={false} connectNulls />
                  {Object.keys(others).map((n, i) => (
                    <Line key={n} type="monotone" dataKey={`val_${n}`} name={`val · ${n}`} stroke={palette[(i + 1) % palette.length]} strokeWidth={1.5} strokeDasharray="5 3" dot={false} isAnimationActive={false} connectNulls />
                  ))}
                  {best && <ReferenceLine x={best.step} stroke="var(--quiet)" strokeDasharray="4 4" label={{ value: `best ${best.val_loss.toFixed(3)} @ ${best.step}`, fill: 'var(--muted)', fontSize: 11, position: 'insideLeft', angle: -90, dx: -8 }} />}
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="flex gap-4 label mt-2">
              <span><span style={{ color: 'var(--series-train)' }}>●</span> train</span>
              <span><span style={{ color: 'var(--series-val)' }}>●</span> val</span>
              {Object.keys(others).map((n, i) => <span key={n}><span style={{ color: palette[(i + 1) % palette.length] }}>╌</span> val · {n}</span>)}
              <span>단위: nat / 토큰 (낮을수록 좋다)</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
