---
title: 도메인 판정 계약
created: 2026-10-29
updated: 2026-10-29
domain: development
status: accepted
---

# 도메인 판정 계약

[브리프](brief.md) 의 목표를 검증 가능한 규칙과 예시로 구체화한 것. 이 프로젝트에서 "맞다"를 판정하는 근거는 세 갈래다:
**모델이 수학적으로 맞게 동작하는가**(테스트), **실험 비교가 공정한가**(규칙), **교재가 독자에게 통하는가**(사용자 판정).
unknown 인 규칙을 에이전트 추측으로 확정하지 않는다. 판정자는 사용자(제작자 본인)다.

두 종류의 eval 을 섞지 않는다: **제품 eval** 은 학습한 GPT 의 품질(val loss·bits/char·생성문, 9장 `src/shllm/eval.py`),
**개발 eval** 은 에이전트 작업의 품질(verify 통과·재작업·근거 없는 완료 주장, [agent-workflow](agent-workflow.md)). 앞의 숫자가 좋아졌다고 작업이 잘 됐다는 뜻은 아니다.

## 규칙과 근거

| ID | 규칙 / 불변조건 | 근거 | 상태 |
|---|---|---|---|
| DOMAIN-001 | v1.0 성공 = 한글 프롬프트를 이어 "말이 되는 문단" 생성 + 모든 장을 내 말로 설명. 합격선은 사용자가 생성문을 보고 선언한다 | brief "성공 기준" | unknown (합격선 미정, todo P1) |
| DOMAIN-002 | 코퍼스는 퍼블릭 도메인 또는 CC 라이선스만. 위키 텍스트는 CC BY-SA 출처 표기(`WIKI-LICENSE.md`)와 함께 | brief "제약", ADR-0003 | confirmed |
| DOMAIN-003 | 학습 데이터는 합본 하나만 (`load_corpus("korean-classics")` 또는 `korean-mixed`). 합본과 `works/*.txt` 를 같이 읽지 않는다 (2배 중복) | CLAUDE.md 함정 | confirmed |
| DOMAIN-004 | 토크나이저가 다른 run 끼리는 토큰당 loss 가 아니라 **글자당 bit** 로 비교한다. 체크포인트는 자기 토크나이저 이름을 기록한다 | 9장, `src/shllm/eval.py:51`, `src/shllm/train.py:122` | confirmed |
| DOMAIN-005 | 모든 학습은 CPU, 이 PC(RAM 7.6GB) 에서 수십 분~수 시간. 넘는 설정은 스케일을 줄이거나 사용자와 상의 | brief "제약" | confirmed |
| DOMAIN-006 | 교재는 LLM 을 처음 배우는 자바 개발자가 읽는다. 장마다 읽기 전에·핵심 용어·손계산 예시·자바 비교·흔한 오해·스스로 점검 | 사용자 피드백 2026-09-16, `docs/writing-guide.md` | confirmed |
| DOMAIN-007 | 교재·노트북·모듈은 번호·제목이 1:1. 노트북은 검증된 코드를 `src/shllm/` 에서 import 한다 | CLAUDE.md "새 장 추가" | confirmed |

상태: confirmed(사용자 확인 또는 이미 운영 중), assumed(임시 가정, 영향 범위를 함께 적음), unknown.

## 수용 예시 (자동 검사와 연결된 것)

| 사례 | 유형 | 기대 결과 | 근거 | 검사 |
|---|---|---|---|---|
| CASE-001 | 정상 | 한글·영문·처음 보는 글자도 encode 뒤 decode 하면 원문 그대로 | 2장 | `tests/test_tokenizer.py` (roundtrip·unseen chars) |
| CASE-002 | 경계 | 생성 중 잘린 BPE 토큰 열도 깨진 글자 없이 앞부분으로 표시 | 2장·8장 | `tests/test_tokenizer.py::test_bpe_truncated_is_prefix` |
| CASE-003 | 실패 금지 | 위치 t 의 출력이 t 이후 토큰에 영향받지 않는다 (인과 마스크) | 5장·6장 | `tests/test_model.py::test_causal_property_end_to_end` |
| CASE-004 | 정상 | 작은 학습이 val loss 를 절반 이하로 줄이고, 체크포인트에 step·토크나이저 이름이 남는다 | 7장, DOMAIN-004 | `tests/test_train.py` |
| CASE-005 | 정상 | 같은 seed 면 같은 생성문, top-k/top-p 가 후보를 잘라낸다 | 8장 | `tests/test_generate.py` |
| CASE-006 | 정상 | 초기화 직후 loss 는 log(V) 근처, perplexity·bits/char 변환이 손계산과 일치 | 9장, DOMAIN-004 | `tests/test_eval.py` |
| CASE-007 | 사용자 판정 | 같은 프롬프트에 대한 small-cpu / mixed-cpu 생성문이 "말이 되는 문단"인가 | DOMAIN-001 | 수동: 노트북 08 · 대시보드 생성 탭 |

## 판정과 변경

- 테스트가 통과해도 DOMAIN-001 이 충족된 것은 아니다. 합격 여부는 CASE-007 을 사용자가 본 뒤에만 기록한다.
- 기대값(예: CASE-004 의 "절반 이하")을 바꿔 테스트를 통과시키지 않는다. 바꿔야 하면 이유를 session-log 에 남긴다.
- unknown 인 DOMAIN-001 에 의존하는 결정(MINOR 승격, v1.0 선언)만 보류하고 나머지 작업은 계속한다.
