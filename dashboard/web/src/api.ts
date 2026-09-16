// FastAPI(dashboard/api/main.py) 와 1:1 대응하는 타입과 호출 함수

export interface RunSummary {
  name: string
  model: Record<string, number | string>
  train: Record<string, number | string>
  n_params: number | null
  steps: number
  elapsed: number
  best_val: number | null
  best_step: number | null
  has_checkpoint: boolean
}
export interface LogRow { step: number; train_loss: number; val_loss: number; lr: number; elapsed: number }
export interface AttentionResult { tokens: string[]; n_layer: number; n_head: number; maps: number[][][][] }
export interface Candidate { token: string; prob: number; raw_prob: number }
export interface GenStep { token: string; id: number; prob: number; raw_prob: number; candidates: Candidate[] }
export interface GenerateResult { prompt_tokens: string[]; steps: GenStep[]; text: string }
export interface GenerateParams {
  run: string
  prompt: string
  max_new_tokens: number
  temperature: number
  top_k: number | null
  top_p: number | null
  repetition_penalty: number
  top_n: number
  seed: number
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}
const post = (url: string, body: unknown) =>
  fetch(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) })

export async function* generateStream(p: GenerateParams): AsyncGenerator<Record<string, unknown>> {
  // NDJSON 스트림을 한 줄(=한 토큰)씩 돌려준다
  const res = await post('/api/generate/stream', p)
  if (!res.ok || !res.body) throw new Error(`${res.status} ${await res.text()}`)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { value, done } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let nl: number
    while ((nl = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, nl).trim()
      buf = buf.slice(nl + 1)
      if (line) yield JSON.parse(line)
    }
  }
}

export const api = {
  runs: () => fetch('/api/runs').then(json<RunSummary[]>),
  log: (run: string) => fetch(`/api/runs/${run}/log`).then(json<LogRow[]>),
  attention: (run: string, text: string) => post('/api/attention', { run, text }).then(json<AttentionResult>),
  generate: (p: GenerateParams) => post('/api/generate', p).then(json<GenerateResult>),
}

export const fmt = {
  loss: (v: number | null | undefined) => (v == null ? ':' : v.toFixed(3)),
  params: (n: number | null) => (n == null ? ':' : `${(n / 1e6).toFixed(1)}M`),
  minutes: (s: number) => (s < 60 ? `${Math.round(s)}s` : `${Math.round(s / 60)}분`),
  pct: (p: number) => `${(p * 100).toFixed(1)}%`,
}
