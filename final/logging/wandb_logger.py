"""Optional Weights & Biases logging for final runs (metrics + live plots)."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_INSTANCE: FinalWandbLogger | None = None
_INSTANCE_LOCK = threading.Lock()


def wandb_enabled() -> bool:
    return os.environ.get("FINAL_WANDB", "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class WandbSettings:
    enabled: bool = False
    project: str = "phylogfn-final"
    entity: str | None = None
    run_name: str | None = None
    group: str | None = None
    tags: tuple[str, ...] = ()

    @classmethod
    def from_env(cls) -> WandbSettings:
        tags_raw = os.environ.get("FINAL_WANDB_TAGS", "")
        tags = tuple(tag.strip() for tag in tags_raw.split(",") if tag.strip())
        return cls(
            enabled=wandb_enabled(),
            project=os.environ.get("WANDB_PROJECT", "phylogfn-final"),
            entity=os.environ.get("WANDB_ENTITY") or None,
            run_name=os.environ.get("WANDB_RUN_NAME") or None,
            group=os.environ.get("WANDB_GROUP") or None,
            tags=tags,
        )

    @classmethod
    def from_cli(
        cls,
        *,
        enabled: bool = False,
        project: str = "phylogfn-final",
        entity: str | None = None,
        run_name: str | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
    ) -> WandbSettings:
        env = cls.from_env()
        return cls(
            enabled=enabled or env.enabled,
            project=project or env.project,
            entity=entity or env.entity,
            run_name=run_name or env.run_name,
            group=group or env.group,
            tags=tuple(tags or env.tags),
        )

    def apply_to_env(self) -> None:
        os.environ["FINAL_WANDB"] = "1" if self.enabled else "0"
        os.environ["WANDB_PROJECT"] = self.project
        if self.entity:
            os.environ["WANDB_ENTITY"] = self.entity
        elif "WANDB_ENTITY" in os.environ:
            del os.environ["WANDB_ENTITY"]
        if self.run_name:
            os.environ["WANDB_RUN_NAME"] = self.run_name
        elif "WANDB_RUN_NAME" in os.environ:
            del os.environ["WANDB_RUN_NAME"]
        if self.group:
            os.environ["WANDB_GROUP"] = self.group
        elif "WANDB_GROUP" in os.environ:
            del os.environ["WANDB_GROUP"]
        if self.tags:
            os.environ["FINAL_WANDB_TAGS"] = ",".join(self.tags)
        elif "FINAL_WANDB_TAGS" in os.environ:
            del os.environ["FINAL_WANDB_TAGS"]


def _scalar_metrics(metrics: dict[str, Any]) -> dict[str, float | int | bool]:
    out: dict[str, float | int | bool] = {}
    for key, value in metrics.items():
        if key.endswith("_hist_counts"):
            continue
        if isinstance(value, bool):
            out[key] = value
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            out[key] = value
    return out


class PlotDirWatcher:
    """Poll plot directories and log new PNGs to wandb as they appear."""

    def __init__(
        self,
        logger: FinalWandbLogger,
        plot_dirs: list[Path],
        *,
        poll_s: float = 1.0,
    ) -> None:
        self.logger = logger
        self.plot_dirs = [Path(path) for path in plot_dirs]
        self.poll_s = poll_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._logged: set[str] = set()

    def __enter__(self) -> PlotDirWatcher:
        if self.logger.active:
            self._thread = threading.Thread(target=self._poll, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.scan_once()

    def scan_once(self) -> None:
        for plot_dir in self.plot_dirs:
            if not plot_dir.exists():
                continue
            for png in sorted(plot_dir.rglob("*.png")):
                key = str(png.resolve())
                if key in self._logged:
                    continue
                self._logged.add(key)
                self.logger.log_plot(png)

    def _poll(self) -> None:
        while not self._stop.is_set():
            self.scan_once()
            self._stop.wait(self.poll_s)


class FinalWandbLogger:
    """Singleton wandb run for a final experiment."""

    def __init__(self, settings: WandbSettings) -> None:
        self.settings = settings
        self._run: Any | None = None
        self._init_lock = threading.Lock()

    @property
    def active(self) -> bool:
        return self.settings.enabled and self._run is not None

    @classmethod
    def configure(cls, settings: WandbSettings) -> FinalWandbLogger | None:
        global _INSTANCE
        if not settings.enabled:
            return None
        with _INSTANCE_LOCK:
            if _INSTANCE is None:
                _INSTANCE = cls(settings)
            else:
                _INSTANCE.settings = settings
            return _INSTANCE

    @classmethod
    def maybe_get(cls) -> FinalWandbLogger | None:
        global _INSTANCE
        with _INSTANCE_LOCK:
            if _INSTANCE is not None:
                return _INSTANCE if _INSTANCE.settings.enabled else None
            settings = WandbSettings.from_env()
            if not settings.enabled:
                return None
            _INSTANCE = cls(settings)
            return _INSTANCE

    def init(self, config: dict[str, Any] | None = None) -> None:
        if not self.settings.enabled or self._run is not None:
            return
        with self._init_lock:
            if self._run is not None:
                return
            import wandb

            run_id = os.environ.get("WANDB_RUN_ID")
            kwargs: dict[str, Any] = {
                "project": self.settings.project,
                "config": config or {},
                "reinit": True,
            }
            if run_id:
                kwargs["id"] = run_id
                kwargs["resume"] = "allow"
            if self.settings.entity:
                kwargs["entity"] = self.settings.entity
            if self.settings.run_name:
                kwargs["name"] = self.settings.run_name
            if self.settings.group:
                kwargs["group"] = self.settings.group
            if self.settings.tags:
                kwargs["tags"] = list(self.settings.tags)
            self._run = wandb.init(**kwargs)
            os.environ["WANDB_RUN_ID"] = self._run.id

    def log_metrics(self, step: int, metrics: dict[str, Any]) -> None:
        if not self.settings.enabled:
            return
        if self._run is None:
            self.init()
        scalars = _scalar_metrics(metrics)
        if not scalars:
            return
        import wandb

        wandb.log(scalars, step=step)

    def log_plot(self, path: Path, *, caption: str | None = None) -> None:
        if not self.settings.enabled:
            return
        if self._run is None:
            self.init()
        if not path.exists():
            return
        import wandb

        key = f"plots/{path.stem}"
        wandb.log({key: wandb.Image(str(path), caption=caption or path.name)})

    def watch_plot_dirs(self, plot_dirs: list[Path]) -> PlotDirWatcher:
        return PlotDirWatcher(self, plot_dirs)

    def finish(self) -> None:
        global _INSTANCE
        if self._run is None:
            return
        import wandb

        wandb.finish()
        self._run = None
        with _INSTANCE_LOCK:
            if _INSTANCE is self:
                _INSTANCE = None
