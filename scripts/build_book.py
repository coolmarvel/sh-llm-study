"""교재(docs/book/NN-*.md) 를 PDF 한 권으로 굽고 바탕화면에 전달한다.

    uv run python scripts/build_book.py            # build/book/ 에 굽고 바탕화면의 이전 판을 지운 뒤 복사
    uv run python scripts/build_book.py --no-copy  # 굽기만

산출물 이름: sh-llm-study-book-v<버전>.pdf (버전 = shllm.__version__). 바탕화면의 같은 접두어 PDF 는
새 판으로 교체된다 (이전 판 삭제) — 항상 최신 한 권만 남긴다.
전달 위치는 환경변수 SHLLM_BOOK_DEST 로 바꿀 수 있다 (기본 /mnt/c/Users/user/Desktop).
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from datetime import date
from pathlib import Path

import markdown
from pygments.formatters import HtmlFormatter
from weasyprint import HTML

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shllm import __version__  # noqa: E402
from shllm.config import REPO_ROOT  # noqa: E402

BOOK_DIR = REPO_ROOT / "docs" / "book"
OUT_DIR = REPO_ROOT / "build" / "book"
DEST_DIR = Path(os.environ.get("SHLLM_BOOK_DEST", "/mnt/c/Users/user/Desktop"))
STEM = "sh-llm-study-book"
FRONTMATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)

CSS = """
@page { size: A4; margin: 22mm 18mm 20mm 18mm;
        @bottom-center { content: counter(page); font-family: 'Noto Sans CJK KR'; font-size: 9pt; color: #666; } }
body { font-family: 'Noto Serif CJK KR', serif; font-size: 10.5pt; line-height: 1.65; color: #111; }
h1, h2, h3 { font-family: 'Noto Sans CJK KR', sans-serif; font-weight: 700; line-height: 1.3; }
h1 { font-size: 20pt; margin: 0 0 14pt; page-break-before: always; }
h2 { font-size: 14pt; margin: 20pt 0 8pt; border-bottom: 1px solid #ccc; padding-bottom: 3pt; }
h3 { font-size: 11.5pt; margin: 14pt 0 6pt; }
p { margin: 0 0 8pt; text-align: justify; }
code, pre { font-family: 'Noto Sans Mono CJK KR', 'DejaVu Sans Mono', monospace; font-size: 8.8pt; }
code { background: #f3f3f3; padding: 0 2pt; border-radius: 2pt; }
pre { background: #f6f6f6; border: 1px solid #ddd; padding: 7pt 9pt; white-space: pre-wrap; line-height: 1.4; margin: 6pt 0 10pt; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9.5pt; }
th, td { border: 1px solid #bbb; padding: 3pt 7pt; vertical-align: top; }
th { background: #eee; font-family: 'Noto Sans CJK KR'; }
blockquote { border-left: 3px solid #bbb; margin: 6pt 0; padding: 2pt 10pt; color: #444; }
ul, ol { margin: 0 0 8pt; padding-left: 20pt; }
li { margin-bottom: 2pt; }
a { color: inherit; text-decoration: none; }
.cover { page-break-before: auto; text-align: center; padding-top: 140pt; }
.cover h1 { page-break-before: auto; font-size: 30pt; border: none; }
.cover p { text-align: center; color: #444; }
.toc { page-break-before: always; }
.toc h1 { page-break-before: auto; }
.toc li { margin-bottom: 4pt; list-style: none; }
"""


def split_frontmatter(text: str) -> tuple[dict[str, str], str]:
    m = FRONTMATTER.match(text)
    if not m:
        return {}, text
    meta = dict(line.split(":", 1) for line in m.group(1).splitlines() if ":" in line)
    return {k.strip(): v.strip() for k, v in meta.items()}, text[m.end() :]


def render_chapter(path: Path) -> tuple[str, str]:
    meta, body = split_frontmatter(path.read_text("utf-8"))
    title = meta.get("title", path.stem)
    html = markdown.markdown(
        body,
        extensions=["tables", "fenced_code", "codehilite", "toc"],
        extension_configs={"codehilite": {"guess_lang": False}},
    )
    return title, html


def build(chapters: list[Path]) -> Path:
    parts = [
        '<div class="cover"><h1>sh-llm-study</h1>'
        "<p>자바 개발자가 밑바닥부터 만드는 소형 한글 GPT — 교재</p>"
        f"<p>v{__version__} · {date.today().isoformat()}</p></div>",
        '<div class="toc"><h1>차례</h1><ul>',
    ]
    bodies = []
    for i, path in enumerate(chapters):
        title, html = render_chapter(path)
        parts.append(f'<li><a href="#ch{i}">{title}</a></li>')
        bodies.append(f'<div id="ch{i}">{html}</div>')
    parts.append("</ul></div>")
    parts.extend(bodies)
    css = CSS + HtmlFormatter(style="default").get_style_defs(".codehilite")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{STEM}-v{__version__}.pdf"
    HTML(string=f"<style>{css}</style>" + "\n".join(parts), base_url=str(BOOK_DIR)).write_pdf(out)
    return out


def deliver(pdf: Path, dest: Path) -> list[Path]:
    """바탕화면의 이전 판(같은 접두어)을 지우고 새 판을 복사한다. 지운 파일 목록을 돌려준다."""
    if not dest.is_dir():
        raise FileNotFoundError(f"전달 위치가 없습니다: {dest} (SHLLM_BOOK_DEST 로 지정)")
    removed = []
    for old in dest.glob(f"{STEM}-*.pdf"):
        old.unlink()
        removed.append(old)
    shutil.copy2(pdf, dest / pdf.name)
    return removed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--no-copy", action="store_true", help="바탕화면 복사 생략")
    ap.add_argument("--dest", type=Path, default=DEST_DIR, help="전달 폴더 (기본: 바탕화면)")
    args = ap.parse_args()

    chapters = sorted(BOOK_DIR.glob("[0-9][0-9]-*.md"))
    if not chapters:
        sys.exit(f"{BOOK_DIR} 에 챕터가 없습니다")
    pdf = build(chapters)
    print(f"교재 {len(chapters)}장 → {pdf} ({pdf.stat().st_size / 1024:.0f} KB)")
    if not args.no_copy:
        removed = deliver(pdf, args.dest)
        for r in removed:
            print(f"  이전 판 삭제: {r.name}")
        print(f"  복사: {args.dest / pdf.name}")


if __name__ == "__main__":
    main()
