"""코퍼스 2차 확장, 한국어 위키백과에서 양질 문서를 받아 data/corpus/wiki/ 와 합본을 만든다 (ADR-0003).

    uv run python scripts/download_wiki.py                    # 알찬 글·좋은 글 + 링크 이웃 → 약 10M 자 (기본)
    uv run python scripts/download_wiki.py --target-chars 4000000 --force

산출물
    data/corpus/wiki/<제목>.txt          문서별 본문 (CC BY-SA 4.0)
    data/corpus/korean-wiki.txt          위키 합본
    data/corpus/korean-mixed.txt         근대문학 합본(korean-classics.txt) + 위키 합본, 2차 학습용
    data/corpus/WIKI-LICENSE.md          출처·라이선스·문서 목록 (저작자 표시 의무)

선택 기준: 커뮤니티가 검수한 "알찬 글"·"좋은 글" 전부 + 그 문서들이 링크하는 문서 중 3,000자 이상. 무작위 문서는 대부분
수백 자짜리 단편이라 쓰지 않는다. 각주·외부 링크·같이 보기 절은 뺀다. 요청 간 0.15초 대기 (API 예절).
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shllm.config import CORPUS_DIR, ensure_data_dirs  # noqa: E402

API = "https://ko.wikipedia.org/w/api.php"
UA = "sh-llm-study/0.1 (learning project; chungmu.xyz@gmail.com)"
SEED_CATEGORIES = ("분류:알찬 글", "분류:좋은 글")
DROP_SECTIONS = {
    "각주",
    "주해",
    "주석",
    "참고 문헌",
    "참고문헌",
    "외부 링크",
    "같이 보기",
    "더 보기",
    "관련 문서",
    "참고 자료",
}
SECTION = re.compile(r"^(=+)\s*(.+?)\s*=+$")
session = requests.Session()
session.headers["User-Agent"] = UA


def api(params: dict) -> dict:
    for attempt in range(4):
        try:
            r = session.get(
                API, params={"format": "json", "formatversion": 2, **params}, timeout=60
            )
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:  # 잠깐의 네트워크 오류는 재시도
            if attempt == 3:
                raise
            print(f"  재시도 {attempt + 1}: {e}")
            time.sleep(2 * (attempt + 1))
    raise RuntimeError("unreachable")


def paged(params: dict, key: str) -> list[dict]:
    out, cont = [], {}
    while True:
        d = api({**params, **cont})
        out += d["query"].get(key, []) if key in d["query"] else [p for p in d["query"]["pages"]]
        if "continue" not in d:
            return out
        cont = d["continue"]


def category_titles(cat: str) -> list[str]:
    return [
        m["title"]
        for m in paged(
            {
                "action": "query",
                "list": "categorymembers",
                "cmtitle": cat,
                "cmnamespace": 0,
                "cmlimit": 500,
            },
            "categorymembers",
        )
    ]


def outgoing_links(title: str) -> list[str]:
    pages = paged(
        {"action": "query", "prop": "links", "plnamespace": 0, "pllimit": 500, "titles": title},
        "pages",
    )
    return [ln["title"] for p in pages for ln in p.get("links", [])]


def extract(title: str) -> str:
    d = api(
        {"action": "query", "prop": "extracts", "explaintext": 1, "titles": title, "redirects": 1}
    )
    return d["query"]["pages"][0].get("extract", "") or ""


def clean(text: str) -> str:
    """절 표식(== 제목 ==)을 한 줄 제목으로, 각주·외부 링크류 절은 통째로 제거, 빈 줄 정리."""
    keep, skipping = [], False
    for line in text.split("\n"):
        m = SECTION.match(line.strip())
        if m:
            depth, name = len(m.group(1)), m.group(2)
            skipping = name in DROP_SECTIONS if depth == 2 else (skipping and depth > 2)
            if not skipping:
                keep.append(name)
            continue
        if not skipping:
            keep.append(line.rstrip())
    out = re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()
    return out + "\n"


def safe_name(title: str) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", title)[:80]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--target-chars", type=int, default=10_000_000, help="위키 합본 목표 글자 수")
    ap.add_argument("--min-chars", type=int, default=3000, help="링크 이웃 문서의 최소 길이")
    ap.add_argument("--force", action="store_true", help="이미 받은 문서도 다시 받는다")
    args = ap.parse_args()

    ensure_data_dirs()
    wiki_dir = CORPUS_DIR / "wiki"
    wiki_dir.mkdir(exist_ok=True)

    seeds: list[str] = []
    for cat in SEED_CATEGORIES:
        titles = category_titles(cat)
        print(f"{cat}: {len(titles)}편")
        seeds += titles
    seeds = list(dict.fromkeys(seeds))  # 순서 유지 중복 제거

    got: dict[str, int] = {}  # 제목 → 글자 수
    total = 0

    def fetch(title: str, min_chars: int) -> bool:
        nonlocal total
        path = wiki_dir / f"{safe_name(title)}.txt"
        if path.exists() and not args.force:
            n = len(path.read_text("utf-8"))
        else:
            text = clean(extract(title))
            time.sleep(0.15)
            if len(text) < min_chars:
                return False
            path.write_text(text, "utf-8")
            n = len(text)
        got[title] = n
        total += n
        return True

    t0 = time.time()
    for i, title in enumerate(seeds):
        fetch(title, min_chars=500)
        if i % 50 == 0:
            print(f"  시드 {i}/{len(seeds)}  누적 {total:,} 자  {time.time() - t0:.0f}s")
    print(f"시드 문서 {len(got)}편, {total:,} 자")

    # 링크 이웃: 시드 문서가 링크하는 문서를 등장 횟수 순으로 (여러 양질 문서가 함께 가리키는 문서일수록 핵심 주제)
    if total < args.target_chars:
        from collections import Counter

        counter: Counter[str] = Counter()
        for i, title in enumerate(seeds):
            for ln in outgoing_links(title):
                if ln not in got and ":" not in ln:
                    counter[ln] += 1
            time.sleep(0.1)
            if i % 50 == 0:
                print(f"  링크 수집 {i}/{len(seeds)}  후보 {len(counter):,}")
        for j, (title, _) in enumerate(counter.most_common()):
            if total >= args.target_chars:
                break
            fetch(title, min_chars=args.min_chars)
            if j % 100 == 0:
                print(f"  이웃 {j}  문서 {len(got)}편  누적 {total:,} 자  {time.time() - t0:.0f}s")

    # 합본
    order = sorted(got)  # 제목순, 재현 가능
    wiki_text = (
        "\n\n".join((wiki_dir / f"{safe_name(t)}.txt").read_text("utf-8").strip() for t in order)
        + "\n"
    )
    (CORPUS_DIR / "korean-wiki.txt").write_text(wiki_text, "utf-8")
    classics = CORPUS_DIR / "korean-classics.txt"
    if classics.exists():
        (CORPUS_DIR / "korean-mixed.txt").write_text(
            classics.read_text("utf-8").strip() + "\n\n" + wiki_text, "utf-8"
        )
    (CORPUS_DIR / "WIKI-LICENSE.md").write_text(
        "# 한국어 위키백과 발췌. CC BY-SA 4.0\n\n"
        "출처: https://ko.wikipedia.org (각 문서의 저자는 해당 문서의 역사 페이지 참조). "
        "라이선스: Creative Commons Attribution-ShareAlike 4.0 (https://creativecommons.org/licenses/by-sa/4.0/deed.ko).\n"
        f"수집: {time.strftime('%Y-%m-%d')}, `scripts/download_wiki.py`, 문서 {len(got)}편, {total:,} 자. "
        "각주·외부 링크·같이 보기 절 제외, 본문만.\n\n## 문서 목록\n\n"
        + "\n".join(f"- {t} ({n:,} 자)" for t, n in sorted(got.items()))
        + "\n",
        "utf-8",
    )
    print(
        f"완료: 위키 {len(got)}편 {total:,} 자 → korean-wiki.txt, korean-mixed.txt ({time.time() - t0:.0f}s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
