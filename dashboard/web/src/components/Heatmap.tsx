// (T, T) 어텐션 가중치 → SVG 격자. 셀 색은 heat-low → heat-high 보간
export default function Heatmap({ matrix, labels, cell = 14, showLabels = true }: { matrix: number[][]; labels: string[]; cell?: number; showLabels?: boolean }) {
  const T = matrix.length
  const pad = showLabels ? Math.min(120, 12 + 7 * Math.max(...labels.map((l) => l.length))) : 0
  const size = T * (cell + 1)
  const color = (v: number) => {
    // #131417 → #9aa3ff
    const lo = [0x13, 0x14, 0x17], hi = [0x9a, 0xa3, 0xff]
    const c = lo.map((l, i) => Math.round(l + (hi[i] - l) * Math.min(1, Math.max(0, v))))
    return `rgb(${c[0]},${c[1]},${c[2]})`
  }
  const show = (s: string) => s.replace(/ /g, '␣').replace(/\n/g, '⏎')
  return (
    <svg width={size + pad} height={size + pad} style={{ display: 'block' }}>
      {matrix.map((row, i) =>
        row.map((v, j) => (
          <rect key={`${i}-${j}`} x={pad + j * (cell + 1)} y={pad + i * (cell + 1)} width={cell} height={cell} fill={color(v)}>
            <title>{`${show(labels[i])} → ${show(labels[j])}: ${v.toFixed(3)}`}</title>
          </rect>
        )),
      )}
      {showLabels &&
        labels.map((l, i) => (
          <g key={i}>
            <text x={pad - 4} y={pad + i * (cell + 1) + cell * 0.8} textAnchor="end" fontSize={10} fill="var(--muted)" fontFamily="var(--font-mono)">{show(l)}</text>
            <text x={pad + i * (cell + 1) + cell * 0.8} y={pad - 4} textAnchor="start" fontSize={10} fill="var(--muted)" fontFamily="var(--font-mono)" transform={`rotate(-90 ${pad + i * (cell + 1) + cell * 0.8} ${pad - 4})`}>{show(l)}</text>
          </g>
        ))}
    </svg>
  )
}
