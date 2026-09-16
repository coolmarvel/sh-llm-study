---
project: sh-llm-study dashboard
reference: linear.app   # ADR-0002: Linear 베이스, 차용 없음 (톤 & 밀도만 계승, 브랜드 색·폰트 자산은 쓰지 않는다)
created: 2026-09-16
updated: 2026-09-16
status: adopted-manual   # omd:init 의 hash-bound 패키지가 아니라 레퍼런스 카탈로그(.claude/data/references/linear.app)에서 손으로 파생. todo 참고
scope: dashboard/ 만. 노트북·교재에는 적용하지 않는다 (CLAUDE.md)
tokens:
  colors:
    canvas: "#0b0c0e"          # 페이지 배경 — Linear 의 #08090a 보다 한 단계 밝게 (긴 손실 곡선을 오래 봐도 눈이 덜 피로)
    surface: "#131417"         # 카드·패널
    surface-raised: "#1a1b1f"  # 호버 · 선택 행
    hairline: "#222327"        # 경계선 (canvas 위 8% 백색)
    foreground: "#f4f5f7"      # 본문
    secondary: "#c3c8d1"       # 보조 텍스트
    muted: "#858a94"           # 라벨·축
    quiet: "#5d6169"           # 비활성
    accent: "#6c74dc"          # 식별 색 하나 — 선택·포커스·현재 run. 남용 금지
    on-accent: "#0b0c0e"
    series-train: "#6c74dc"    # 손실 곡선 train
    series-val: "#e0a458"      # 손실 곡선 val (accent 와 색상환 반대편, 색약 구분 가능)
    heat-low: "#131417"        # 어텐션 히트맵 0
    heat-high: "#9aa3ff"       # 어텐션 히트맵 1
    prob-bar: "#3f4560"        # 확률 막대 (뽑힌 토큰은 accent)
    danger: "#e5484d"
  typography:
    family:
      ui: "Inter Variable, Pretendard Variable, system-ui, sans-serif"   # 한글은 Pretendard 로 폴백 (Inter 에 한글 없음)
      mono: "JetBrains Mono, D2Coding, ui-monospace, monospace"           # 토큰·숫자·로그
    heading: { size: 20, weight: 590, lineHeight: 1.3, tracking: -0.2 }
    title:   { size: 15, weight: 510, lineHeight: 1.4, tracking: -0.1 }
    body:    { size: 13, weight: 400, lineHeight: 1.55, tracking: 0 }
    label:   { size: 12, weight: 400, lineHeight: 1.5, tracking: 0 }    # 축·범례·메타
    mono:    { size: 12.5, weight: 400, lineHeight: 1.6 }
  spacing: { xs: 4, sm: 8, md: 12, lg: 16, xl: 24, xxl: 32 }
  rounded: { control: 6, card: 8, pill: 9999 }
  shadow:
    none: "none"                                   # 다크 캔버스에서는 그림자 대신 hairline 으로 층을 구분
  motion:
    fast: "120ms ease-out"                         # 호버·선택
    normal: "200ms ease-out"                       # 패널 전환
    none-for-data: true                            # 차트 데이터 갱신은 애니메이션 없음 (숫자를 읽는 중에 움직이면 안 된다)
  components:
    app-shell:      { type: layout, sidebar: 220, header: 44, content-max: 1280, gutter: 24 }
    nav-item:       { type: button, bg: transparent, fg: muted, radius: control, height: 32, padding: "0 12px", font: body, states: "hover → surface-raised + secondary, selected → surface-raised + foreground + 좌측 2px accent" }
    card:           { type: card, bg: surface, border: "1px solid hairline", radius: card, padding: 16, title: title }
    primary-action: { type: button, bg: foreground, fg: canvas, radius: pill, height: 32, padding: "0 14px", font: "13 / 510", states: "hover 92% opacity, disabled 40%" }
    ghost-action:   { type: button, bg: transparent, fg: secondary, border: "1px solid hairline", radius: pill, height: 32, padding: "0 12px", states: "hover → surface-raised" }
    field:          { type: input, bg: canvas, fg: foreground, border: "1px solid hairline", radius: control, height: 32, padding: "0 10px", font: mono, states: "focus → border accent" }
    slider-field:   { type: input, label: label, value: mono, track: hairline, thumb: accent }
    stat:           { type: text, label: label(muted), value: "mono 18 / 510 (foreground)" }
    run-row:        { type: row, height: 36, padding: "0 12px", cols: "이름 · 파라미터 · step · best val · 경과", states: "selected → surface-raised + accent 점" }
    chart:          { type: chart, bg: transparent, grid: hairline, axis: label(muted), series: "train 2px, val 2px, best 점선 quiet", tooltip: "surface + hairline, mono" }
    heatmap:        { type: chart, cell-min: 14, gap: 1, scale: "heat-low → heat-high", labels: "mono, 회전 90" }
    token-chip:     { type: chip, bg: surface-raised, fg: foreground, radius: control, padding: "2px 6px", font: mono, states: "hover → 확률 막대 표시" }
---

# DESIGN.md — sh-llm-study 대시보드

## 원칙

1. **데이터가 주인공.** 손실 곡선·어텐션 맵·토큰 확률이 화면의 80% 를 차지한다. 장식·배너·일러스트 없음.
2. **어두운 캔버스, 좁은 명도 계단.** Linear 의 톤: 배경은 거의 검정, 텍스트는 거의 흰색, 그 사이 3단계(secondary·muted·quiet)로 정보 위계를 만든다. 색은 accent 하나 + 시리즈 색 둘뿐이다.
3. **밀도 높은 개발자 도구.** 13px 본문, 32px 컨트롤, 16px 카드 패딩. 여백으로 고급스러움을 내지 않는다.
4. **숫자는 모노스페이스.** loss·step·확률·토큰은 항상 mono. 자릿수가 흔들리지 않게 소수 셋째 자리 고정.
5. **움직이지 않는다.** 데이터 갱신에 애니메이션 없음. 호버·선택만 120ms.
6. **한글 우선.** UI 문구는 한국어, 기술 용어(loss, step, top-k)는 영문 그대로. Inter 에 한글이 없으므로 Pretendard 폴백을 반드시 선언.

## 화면

| 화면 | 목적 | 구성 |
|---|---|---|
| 실험 (`/`) | run 목록과 손실 곡선 | 좌: run-row 목록 · 우: chart(train/val, best 점선) + stat 4개(파라미터·step·best val·경과) |
| 어텐션 (`/attention`) | 문장을 넣고 층·헤드별 시선 보기 | 상: field(문장) + run 선택 · 하: heatmap 격자(층 × 헤드), 셀 클릭 시 확대 |
| 생성 (`/generate`) | 토큰별 확률을 보며 생성 | 좌: field(프롬프트) + slider-field(temperature·top-k·top-p·반복 억제) + primary-action · 우: 생성문(token-chip 나열) + 선택 토큰의 상위 10 확률 막대 |

## 금지

- accent 를 배경 면으로 넓게 칠하기 (버튼 fill 은 foreground, accent 는 선·점·선택 표시)
- 그림자로 층 만들기 (hairline 으로)
- 그라데이션, 이모지 아이콘, 둥근 큰 카드(라운드 8 초과)
- 차트 기본 팔레트 그대로 쓰기 (series-train / series-val 만)

## 근거

- ADR-0002 (스택·디자인 레퍼런스), `.claude/data/references/linear.app/DESIGN.md` (톤 출처, 2026-07-12 검증본)
- 색·크기 값은 레퍼런스에서 **파생**한 자체 값이며 Linear 의 자산(Inter Variable·Berkeley Mono 웹폰트, 로고, 인디고 브랜드)을 쓰지 않는다.
