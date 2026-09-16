import { useEffect, useState } from 'react'

// 교재 '관련 코드' 의 파일 경로를 눌렀을 때 오른쪽에 뜨는 소스 뷰어 (/api/source)
export default function SourcePanel({ path, onClose }: { path: string; onClose: () => void }) {
  const [doc, setDoc] = useState<{ path: string; lines: number; html: string; css: string } | null>(null)
  const [error, setError] = useState<string | null>(null)
  useEffect(() => {
    setDoc(null); setError(null)
    fetch(`/api/source?path=${encodeURIComponent(path)}`).then(async (r) => { if (!r.ok) throw new Error(await r.text()); return r.json() }).then(setDoc).catch((e) => setError(String(e)))
  }, [path])
  return (
    <div className="card source-panel">
      <div className="flex items-center gap-3 mb-2">
        <span className="mono" style={{ color: 'var(--fg)' }}>{path}</span>
        {doc && <span className="label">{doc.lines} 줄</span>}
        <button className="btn-ghost ml-auto" onClick={onClose}>닫기</button>
      </div>
      {error && <div style={{ color: 'var(--danger)' }}>{error}</div>}
      {doc && (<><style>{doc.css}</style><div className="source-body" dangerouslySetInnerHTML={{ __html: doc.html }} /></>)}
    </div>
  )
}
