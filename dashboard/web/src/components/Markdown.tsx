import { marked } from 'marked'
import { useMemo } from 'react'

// 노트북 markdown 셀과 교재의 짧은 텍스트를 HTML 로. 교재 본문은 서버(markdown+pygments)가 렌더한다
marked.setOptions({ gfm: true, breaks: false })
export default function Markdown({ text, className = 'prose-dark' }: { text: string; className?: string }) {
  const html = useMemo(() => marked.parse(text) as string, [text])
  return <div className={className} dangerouslySetInnerHTML={{ __html: html }} />
}
