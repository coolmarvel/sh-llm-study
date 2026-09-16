"""notebooks/*.ipynb 를 저장소 표준 형태로 맞춘다 (VS Code 가 열자마자 "변경됨" 을 만들지 않게).

    uv run python scripts/normalize_notebooks.py           # 제자리 수정
    uv run python scripts/normalize_notebooks.py --check   # 어긋난 파일이 있으면 exit 1 (verify.sh 가 호출)

표준: kernelspec 은 VS Code Jupyter 확장이 .venv 커널을 고를 때 쓰는 값과 같게, language_info 포함,
셀 id 는 순번("0","1",…), git 의 nbstripout 필터가 커밋본에 쓰는 것과 같은 규칙, 출력·실행 번호 없음.
근거: 2026-09-16 VS Code 에서 03 노트북이 열 때마다 dirty 가 되던 사고 (docs/session-log.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nbformat

REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_DIR = REPO_ROOT / "notebooks"
METADATA = {
    "kernelspec": {
        "display_name": "sh-llm-study (3.12.3)",
        "language": "python",
        "name": "python3",
    },
    "language_info": {"name": "python", "version": "3.12.3"},
}


def normalize(nb: nbformat.NotebookNode) -> nbformat.NotebookNode:
    nb.metadata = nbformat.NotebookNode(METADATA)
    nb.nbformat, nb.nbformat_minor = 4, 5
    for i, cell in enumerate(nb.cells):
        cell.id = str(i)
        cell.metadata = nbformat.NotebookNode()
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    return nb


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--check", action="store_true", help="수정하지 않고 어긋난 파일만 보고")
    args = ap.parse_args()

    dirty: list[Path] = []
    for path in sorted(NOTEBOOK_DIR.glob("*.ipynb")):
        original = path.read_text("utf-8")
        nb = normalize(nbformat.read(path, as_version=4))
        rendered = nbformat.writes(nb) + "\n"
        if rendered != original:
            dirty.append(path)
            if not args.check:
                path.write_text(rendered, "utf-8")
    if args.check and dirty:
        print("정규화되지 않은 노트북:", *[p.name for p in dirty])
        print("→ uv run python scripts/normalize_notebooks.py")
        sys.exit(1)
    print(f"{'어긋난' if args.check else '수정한'} 노트북 {len(dirty)}개")


if __name__ == "__main__":
    main()
