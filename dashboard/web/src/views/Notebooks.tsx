import { useEffect, useRef, useState } from 'react'
import CodeMirror from '@uiw/react-codemirror'
import { python } from '@codemirror/lang-python'
import { oneDark } from '@codemirror/theme-one-dark'
import Markdown from '../components/Markdown'

interface NB { name: string; title: string; cells: number; executed: boolean; executed_at: number | null }
interface Output { output_type: string; name?: string; text?: string | string[]; data?: Record<string, string | string[]>; ename?: string; evalue?: string; traceback?: string[]; state?: string; execution_count?: number | null }
interface Cell { id: string; cell_type: 'markdown' | 'code'; source: string; outputs: Output[]; running?: boolean; count?: number | null; editing?: boolean }

const join = (t: string | string[] | undefined) => (Array.isArray(t) ? t.join('') : (t ?? ''))

function OutputView({ o }: { o: Output }) {
  if (o.output_type === 'stream') return <pre className={`out ${o.name === 'stderr' ? 'stderr' : ''}`}>{join(o.text)}</pre>
  if (o.output_type === 'error') return <pre className="out error">{o.ename}: {o.evalue}{'\n'}{(o.traceback ?? []).join('\n')}</pre>
  if (o.data) {
    if (o.data['image/png']) return <img className="out-img" src={`data:image/png;base64,${join(o.data['image/png'])}`} alt="" />
    if (o.data['image/svg+xml']) return <div className="out-img" dangerouslySetInnerHTML={{ __html: join(o.data['image/svg+xml']) }} />
    if (o.data['text/html']) return <div className="out-html" dangerouslySetInnerHTML={{ __html: join(o.data['text/html']) }} />
    if (o.data['text/plain']) return <pre className="out">{join(o.data['text/plain'])}</pre>
  }
  return null
}

export default function Notebooks() {
  const [list, setList] = useState<NB[]>([])
  const [current, setCurrent] = useState<string | null>(() => location.hash.split('/')[2] || null)
  const [cells, setCells] = useState<Cell[]>([])
  const [kernel, setKernel] = useState<'off' | 'starting' | 'idle' | 'busy'>('off')
  const [dirty, setDirty] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const stopRef = useRef(false)

  useEffect(() => {
    fetch('/api/notebooks').then((r) => r.json()).then((l: NB[]) => { setList(l); if (!current && l.length) setCurrent(l[0].name) }).catch((e) => setError(String(e)))
  }, [])
  useEffect(() => {
    if (!current) return
    location.hash = `#/notebooks/${current}`
    fetch(`/api/notebooks/${current}/cells`).then((r) => r.json()).then((d) => { setCells(d.cells); setKernel(d.kernel_alive ? 'idle' : 'off'); setDirty(false) }).catch((e) => setError(String(e)))
  }, [current])

  const update = (id: string, patch: Partial<Cell>) => setCells((cs) => cs.map((c) => (c.id === id ? { ...c, ...patch } : c)))

  const runCell = async (cell: Cell) => {
    if (cell.cell_type !== 'code' || !current) return
    if (kernel === 'off') setKernel('starting')
    update(cell.id, { outputs: [], running: true })
    try {
      const res = await fetch(`/api/notebooks/${current}/execute`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ code: cell.source }) })
      if (!res.ok || !res.body) throw new Error(`${res.status} ${await res.text()}`)
      setKernel('busy')
      const reader = res.body.getReader(); const dec = new TextDecoder(); let buf = ''
      for (;;) {
        const { value, done } = await reader.read(); if (done) break
        buf += dec.decode(value, { stream: true })
        let nl: number
        while ((nl = buf.indexOf('\n')) >= 0) {
          const line = buf.slice(0, nl).trim(); buf = buf.slice(nl + 1)
          if (!line) continue
          const o: Output = JSON.parse(line)
          if (o.output_type === 'status') { if (o.state === 'idle') update(cell.id, { count: o.execution_count ?? null }); continue }
          setCells((cs) => cs.map((c) => {
            if (c.id !== cell.id) return c
            const last = c.outputs[c.outputs.length - 1]
            if (o.output_type === 'stream' && last?.output_type === 'stream' && last.name === o.name) return { ...c, outputs: [...c.outputs.slice(0, -1), { ...last, text: join(last.text) + join(o.text) }] }
            return { ...c, outputs: [...c.outputs, o] }
          }))
        }
      }
    } catch (e) { setError(String(e)) } finally { update(cell.id, { running: false }); setKernel('idle') }
  }
  const runAll = async () => {
    stopRef.current = false
    for (const c of cells) { if (stopRef.current) break; if (c.cell_type === 'code') await runCell({ ...c, source: cells.find((x) => x.id === c.id)?.source ?? c.source }) }
  }
  const kernelAction = async (a: string) => {
    if (!current) return
    if (a === 'interrupt') stopRef.current = true
    const r = await fetch(`/api/notebooks/${current}/kernel/${a}`, { method: 'POST' }).then((r) => r.json())
    setKernel(r.alive ? 'idle' : 'off')
    if (a === 'restart' || a === 'shutdown') setCells((cs) => cs.map((c) => ({ ...c, count: null })))
  }
  const save = async () => {
    if (!current) return
    await fetch(`/api/notebooks/${current}/cells`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ cells: cells.map(({ id, cell_type, source }) => ({ id, cell_type, source })) }) })
    setDirty(false)
  }
  const addCell = (after: number, type: 'code' | 'markdown') => {
    const c: Cell = { id: `new-${Date.now()}`, cell_type: type, source: '', outputs: [], editing: true }
    setCells((cs) => [...cs.slice(0, after + 1), c, ...cs.slice(after + 1)]); setDirty(true)
  }
  const removeCell = (id: string) => { setCells((cs) => cs.filter((c) => c.id !== id)); setDirty(true) }
  const kernelLabel = { off: '커널 꺼짐 (첫 실행 때 켜집니다)', starting: '커널 시작 중…', idle: '커널 대기', busy: '실행 중' }[kernel]

  return (
    <div>
      <div className="flex items-baseline justify-between mb-4">
        <h1 className="h1">노트북</h1>
        <span className="label">셀을 고쳐 실행할 수 있습니다 · 저장하면 노트북 파일에 반영 · 상태는 커널을 재시작할 때까지 유지</span>
      </div>
      {error && <div className="card mb-4" style={{ borderColor: 'var(--danger)' }}>{error} <button className="btn-ghost ml-2" onClick={() => setError(null)}>닫기</button></div>}
      <div className="grid gap-4 two-col" style={{ gridTemplateColumns: '280px 1fr' }}>
        <aside className="card" style={{ padding: 8, alignSelf: 'start', position: 'sticky', top: 20 }}>
          {list.map((n) => (
            <a key={n.name} href={`#/notebooks/${n.name}`} onClick={(e) => { e.preventDefault(); setCurrent(n.name) }} className={`nav-item ${n.name === current ? 'selected' : ''}`} style={{ height: 'auto', padding: '6px 10px', lineHeight: 1.35 }}>
              <span style={{ fontSize: 12.5 }}>{n.title.replace(/ \(실습\)$/, '')}</span>
            </a>
          ))}
        </aside>
        <div className="flex flex-col gap-3">
          <div className="card toolbar">
            <span className="mono">{current}.ipynb</span>
            <span className={`kernel-dot ${kernel}`} /><span className="label">{kernelLabel}</span>
            <div className="ml-auto flex gap-2">
              <button className="btn-ghost" onClick={runAll} disabled={kernel === 'busy'}>모두 실행</button>
              <button className="btn-ghost" onClick={() => kernelAction('interrupt')} disabled={kernel !== 'busy'}>중단</button>
              <button className="btn-ghost" onClick={() => kernelAction('restart')}>커널 재시작</button>
              <button className="btn-ghost" onClick={() => kernelAction('shutdown')} disabled={kernel === 'off'}>커널 끄기</button>
              <button className="btn-primary" onClick={save} disabled={!dirty}>{dirty ? '저장' : '저장됨'}</button>
            </div>
          </div>
          {cells.map((c, i) => (
            <div key={c.id} className={`cell ${c.cell_type} ${c.running ? 'running' : ''}`}>
              <div className="cell-gutter">
                {c.cell_type === 'code' ? (
                  <button className="run-btn" title="이 셀 실행 (Shift+Enter)" onClick={() => runCell(c)} disabled={kernel === 'busy' && !c.running}>{c.running ? '■' : '▶'}</button>
                ) : (
                  <button className="run-btn" title="편집" onClick={() => update(c.id, { editing: !c.editing })}>{c.editing ? '✓' : '✎'}</button>
                )}
                <span className="mono label">{c.cell_type === 'code' ? `[${c.count ?? ' '}]` : 'md'}</span>
              </div>
              <div className="cell-body">
                {c.cell_type === 'code' || c.editing ? (
                  <CodeMirror value={c.source} theme={oneDark} extensions={c.cell_type === 'code' ? [python()] : []} basicSetup={{ lineNumbers: false, foldGutter: false, highlightActiveLine: false }}
                    onChange={(v) => { update(c.id, { source: v }); setDirty(true) }}
                    onKeyDown={(e) => { if (e.shiftKey && e.key === 'Enter') { e.preventDefault(); if (c.cell_type === 'code') runCell(c); else update(c.id, { editing: false }) } }} />
                ) : (
                  <div onDoubleClick={() => update(c.id, { editing: true })}><Markdown text={c.source} /></div>
                )}
                {c.outputs.length > 0 && <div className="outputs">{c.outputs.map((o, j) => <OutputView key={j} o={o} />)}</div>}
              </div>
              <div className="cell-actions">
                <button title="아래에 코드 셀" onClick={() => addCell(i, 'code')}>+ 코드</button>
                <button title="아래에 설명 셀" onClick={() => addCell(i, 'markdown')}>+ 설명</button>
                <button title="셀 삭제" onClick={() => removeCell(c.id)}>삭제</button>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
