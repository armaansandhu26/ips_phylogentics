"""Wait for run directories and execute subprocess commands."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

from final.methods.base import CommandSpec

if TYPE_CHECKING:
    from final.logging.wandb_logger import FinalWandbLogger


def wait_for_run_dir(
    output_root: Path,
    run_name: str,
    *,
    marker_name: str,
    glob_pattern: str | None = None,
    excluded: set[Path] | None = None,
    timeout_s: int = 300,
) -> Path:
    excluded = {path.resolve() for path in (excluded or set())}
    pattern = glob_pattern or f"*_{run_name}_*"
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        matches = sorted(
            path
            for path in output_root.glob(pattern)
            if path.is_dir() and path.resolve() not in excluded
        )
        if matches and (matches[-1] / marker_name).exists():
            return matches[-1]
        time.sleep(5)
    raise TimeoutError(
        f"run directory for {run_name} not created within {timeout_s}s under {output_root}"
    )


def run_command(
    spec: CommandSpec,
    *,
    log_file: Path | None = None,
    plot_dirs: list[Path] | None = None,
    wandb_logger: FinalWandbLogger | None = None,
) -> None:
    printable = " ".join(spec.argv)
    print(f"[final] running: {printable}", flush=True)
    watcher = None
    if wandb_logger is not None and plot_dirs:
        watcher = wandb_logger.watch_plot_dirs(plot_dirs)
        watcher.__enter__()

    try:
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with log_file.open("a", encoding="utf-8") as handle:
                handle.write(f"\n[final] {time.strftime('%F %T')} {printable}\n")
                handle.flush()
                subprocess.run(
                    spec.argv,
                    cwd=spec.cwd,
                    env=spec.env,
                    check=True,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                )
            return
        subprocess.run(spec.argv, cwd=spec.cwd, env=spec.env, check=True)
    finally:
        if watcher is not None:
            watcher.__exit__(None, None, None)


def popen_command(spec: CommandSpec, *, log_file: Path | None = None) -> subprocess.Popen:
    printable = " ".join(spec.argv)
    print(f"[final] launching: {printable}", flush=True)
    stdout = log_file.open("a") if log_file is not None else subprocess.DEVNULL
    return subprocess.Popen(
        spec.argv,
        cwd=spec.cwd,
        env=spec.env,
        stdout=stdout,
        stderr=subprocess.STDOUT,
    )
