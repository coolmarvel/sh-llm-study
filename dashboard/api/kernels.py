"""웹 내장 노트북 실행기, 노트북마다 IPython 커널 하나를 띄워 셀을 실행하고 출력을 스트리밍한다.

JupyterLab 서버 없이 대시보드 안에서 노트북을 돌리기 위한 최소 구현 (jupyter_client 직접 사용).
    KernelPool.get(name)          이름(노트북 stem)별 커널. 없으면 notebooks/ 를 작업 폴더로 시작
    KernelPool.execute(name, code) 실행하며 출력 dict 를 하나씩 yield (stream · display_data · error · status)
    KernelPool.interrupt / restart / shutdown
출력 형식은 nbformat 의 output 과 같게 맞춰 프론트가 그대로 그린다: {"output_type": "stream", "name": "stdout", "text": ...},
{"output_type": "display_data"|"execute_result", "data": {"text/plain": ..., "image/png": base64}}, {"output_type": "error", "ename", "evalue", "traceback"}.
"""

from __future__ import annotations

import os
import queue
import re
import threading
from collections.abc import Iterator
from pathlib import Path

from jupyter_client import KernelManager

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class KernelPool:
    def __init__(self, cwd: Path, pythonpath: Path) -> None:
        self.cwd = cwd
        self.pythonpath = pythonpath
        self._kernels: dict[str, tuple[KernelManager, object]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def names(self) -> list[str]:
        return sorted(self._kernels)

    def get(self, name: str):
        with self._guard:
            if name not in self._kernels:
                env = dict(os.environ)
                env["PYTHONPATH"] = str(self.pythonpath) + os.pathsep + env.get("PYTHONPATH", "")
                env.setdefault("MPLBACKEND", "module://matplotlib_inline.backend_inline")
                km = KernelManager(kernel_name="python3")
                km.start_kernel(cwd=str(self.cwd), env=env)
                kc = km.client()
                kc.start_channels()
                kc.wait_for_ready(timeout=60)
                self._kernels[name] = (km, kc)
                self._locks[name] = threading.Lock()
            return self._kernels[name]

    def execute(self, name: str, code: str, timeout: float = 1800) -> Iterator[dict]:
        km, kc = self.get(name)
        lock = self._locks[name]
        if not lock.acquire(blocking=False):
            yield {
                "output_type": "error",
                "ename": "Busy",
                "evalue": "이 노트북의 커널이 다른 셀을 실행 중입니다",
                "traceback": [],
            }
            return
        try:
            msg_id = kc.execute(code, store_history=True)
            yield {"output_type": "status", "state": "running"}
            while True:
                try:
                    msg = kc.get_iopub_msg(timeout=timeout)
                except queue.Empty:
                    yield {
                        "output_type": "error",
                        "ename": "Timeout",
                        "evalue": f"{timeout}s 안에 끝나지 않았습니다",
                        "traceback": [],
                    }
                    return
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                t, c = msg["msg_type"], msg["content"]
                if t == "stream":
                    yield {"output_type": "stream", "name": c["name"], "text": c["text"]}
                elif t in ("display_data", "execute_result"):
                    yield {
                        "output_type": t,
                        "data": {
                            k: v
                            for k, v in c["data"].items()
                            if k in ("text/plain", "text/html", "image/png", "image/svg+xml")
                        },
                    }
                elif t == "error":
                    yield {
                        "output_type": "error",
                        "ename": c["ename"],
                        "evalue": c["evalue"],
                        "traceback": [ANSI.sub("", ln) for ln in c["traceback"]],
                    }
                elif t == "status" and c["execution_state"] == "idle":
                    break
            # 실행 번호, 셸 채널에서 이 실행의 응답을 찾는다 (다른 응답이 섞여 있을 수 있다)
            count = None
            try:
                for _ in range(20):
                    reply = kc.get_shell_msg(timeout=10)
                    if reply["parent_header"].get("msg_id") == msg_id:
                        count = reply["content"].get("execution_count")
                        break
            except queue.Empty:
                pass
            yield {"output_type": "status", "state": "idle", "execution_count": count}
        finally:
            lock.release()

    def interrupt(self, name: str) -> None:
        if name in self._kernels:
            self._kernels[name][0].interrupt_kernel()

    def restart(self, name: str) -> None:
        if name in self._kernels:
            km, kc = self._kernels[name]
            km.restart_kernel(now=True)
            kc.wait_for_ready(timeout=60)

    def shutdown(self, name: str | None = None) -> None:
        with self._guard:
            targets = [name] if name else list(self._kernels)
            for n in targets:
                if n in self._kernels:
                    km, kc = self._kernels.pop(n)
                    self._locks.pop(n, None)
                    kc.stop_channels()
                    km.shutdown_kernel(now=True)
