"""한글 퍼블릭 도메인 코퍼스 다운로드 — ko.wikisource.org 의 저작권 만료 단편 소설.

저작권: 한국 저작권법상 저자 사후 70년(2013년 이전에 이미 50년이 지난 저작물은 만료 유지).
아래 작가는 모두 만료됐다: 김유정(1937) 현진건(1943) 이상(1937) 이효석(1942) 나도향(1926) 김동인(1951) 채만식(1950).
위키문헌 본문은 CC BY-SA 4.0 / 퍼블릭 도메인 — 학습용으로 자유롭게 쓸 수 있다.

사용법:
    uv run python scripts/download_corpus.py            # data/corpus/works/*.txt + 합본 data/corpus/korean-classics.txt
    uv run python scripts/download_corpus.py --list     # 대상 목록만 출력
"""

from __future__ import annotations

import argparse
import re
import sys
import time

import requests

from shllm.config import CORPUS_DIR, ensure_data_dirs

API = "https://ko.wikisource.org/w/api.php"
UA = "sh-llm-study/0.1 (learning project; chungmu.xyz@gmail.com)"

# (검색어, 저장 파일명). 위키문헌 페이지 제목은 바뀔 수 있어 검색 API 로 찾는다.
WORKS: list[tuple[str, str]] = [
    ("봄봄 김유정", "kim-yujeong-bombom"),
    ("동백꽃 김유정", "kim-yujeong-dongbaekkkot"),
    ("소낙비 김유정", "kim-yujeong-sonakbi"),
    ("만무방 김유정", "kim-yujeong-manmubang"),
    ("운수 좋은 날 현진건", "hyun-jingeon-unsu-joeun-nal"),
    ("빈처 현진건", "hyun-jingeon-bincheo"),
    ("B사감과 러브레터 현진건", "hyun-jingeon-b-sagam"),
    ("술 권하는 사회 현진건", "hyun-jingeon-sul-gwonhaneun-sahoe"),
    ("날개 이상", "yi-sang-nalgae"),
    ("메밀꽃 필 무렵 이효석", "lee-hyoseok-memilkkot"),
    ("벙어리 삼룡이 나도향", "na-dohyang-samryongi"),
    ("물레방아 나도향", "na-dohyang-mullebanga"),
    ("감자 김동인", "kim-dongin-gamja"),
    ("배따라기 김동인", "kim-dongin-baettaragi"),
    ("광염 소나타 김동인", "kim-dongin-gwangyeom-sonata"),
    ("레디메이드 인생 채만식", "chae-manshik-readymade"),
    ("치숙 채만식", "chae-manshik-chisuk"),
    ("탁류 채만식", "chae-manshik-takryu"),
    ("태평천하 채만식", "chae-manshik-taepyeongcheonha"),
    ("무정 이광수", "lee-gwangsu-mujeong"),
]


def api(params: dict) -> dict:
    params = {"format": "json", "formatversion": 2, **params}
    r = requests.get(API, params=params, headers={"User-Agent": UA}, timeout=30)
    r.raise_for_status()
    return r.json()


def search_title(query: str) -> str | None:
    hits = (
        api({"action": "query", "list": "search", "srsearch": query, "srlimit": 5})
        .get("query", {})
        .get("search", [])
    )
    for h in hits:
        title = h["title"]
        # 작가 페이지·목차·번역 등은 건너뛴다
        if any(bad in title for bad in ("저자:", "토론:", "분류:", "번역:")):
            continue
        return title
    return None


def fetch_text(title: str) -> str:
    """페이지 텍스트를 평문으로. 여러 장(章)으로 쪼개진 작품은 하위 페이지를 이어 붙인다."""
    data = api({"action": "parse", "page": title, "prop": "text|links", "disablelimitreport": 1})
    if "error" in data:
        return ""
    html = data["parse"]["text"]
    text = html_to_text(html)
    # 하위 페이지(예: "탁류/1")가 있으면 순서대로 붙인다
    subs = sorted(
        (
            ln["title"]
            for ln in data["parse"].get("links", [])
            if ln["title"].startswith(title + "/")
        ),
        key=lambda t: [int(x) if x.isdigit() else x for x in re.split(r"(\d+)", t)],
    )
    for sub in subs:
        time.sleep(0.3)
        sd = api({"action": "parse", "page": sub, "prop": "text", "disablelimitreport": 1})
        if "error" not in sd:
            text += "\n\n" + html_to_text(sd["parse"]["text"])
    return text


def html_to_text(html: str) -> str:
    html = re.sub(r"<(script|style|table)[^>]*>.*?</\1>", " ", html, flags=re.S)
    html = re.sub(
        r"<div class=\"(ws-noexport|licenseContainer|headertemplate)[^\"]*\".*?</div>",
        " ",
        html,
        flags=re.S,
    )
    html = re.sub(r"<br\s*/?>", "\n", html)
    html = re.sub(r"</(p|div|h\d|li)>", "\n", html)
    text = re.sub(r"<[^>]+>", "", html)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
    )
    text = re.sub(r"\[\d+\]", "", text)  # 각주 번호
    text = text.replace("&#160;", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = drop_boilerplate(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# 위키문헌 페이지의 머리말(이전/다음 링크·저자·자매 프로젝트)과 꼬리말(라이선스 상자)에만 나오는 문구
BOILERPLATE = (
    "저자:",
    "자매 프로젝트",
    "위키백과",
    "자료가 있습니다",
    "[편집]",
    "출전:",
    "Public domain",
    "{{PD",
    "이 저작물은 저자가 사망한",
    "이 저작물이 미국에서도",
    "1931년에서 1977년 사이에",
)


def drop_boilerplate(text: str) -> str:
    keep = []
    for line in text.split("\n"):
        s = line.strip()
        if s in ("←", "→", "라이선스", "목차"):
            continue
        if any(b in s for b in BOILERPLATE):
            continue
        keep.append(line.rstrip())
    return "\n".join(keep)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--force", action="store_true", help="이미 받은 파일도 다시 받는다")
    args = parser.parse_args()
    if args.list:
        for q, name in WORKS:
            print(f"{name:40s} <- {q}")
        return 0

    ensure_data_dirs()
    works_dir = CORPUS_DIR / "works"
    works_dir.mkdir(exist_ok=True)
    combined: list[str] = []
    for query, name in WORKS:
        out = works_dir / f"{name}.txt"
        if out.exists() and not args.force:
            print(f"skip  {name} (이미 있음, {out.stat().st_size:,} bytes)")
            combined.append(out.read_text("utf-8"))
            continue
        title = search_title(query)
        if not title:
            print(f"MISS  {name}: '{query}' 검색 결과 없음", file=sys.stderr)
            continue
        text = fetch_text(title)
        if len(text) < 2000:
            print(f"MISS  {name}: '{title}' 본문이 너무 짧음 ({len(text)} chars)", file=sys.stderr)
            continue
        out.write_text(text, "utf-8")
        combined.append(text)
        print(f"ok    {name} <- '{title}' ({len(text):,} chars)")
        time.sleep(0.5)

    if combined:
        all_path = CORPUS_DIR / "korean-classics.txt"
        all_path.write_text("\n\n".join(combined), "utf-8")
        print(
            f"\n합본 {all_path} ({all_path.stat().st_size:,} bytes, {sum(map(len, combined)):,} chars)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
