"""Checkpoint resume helpers for grpo_experiments training runs."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from grpo_experiments.metrics import OutcomeTracker


TRAINING_STATE_FILE = "training_state.json"
TRAINER_STATE_FILE = "trainer_state.pt"


def is_on_policy_training_mode(training_mode: str) -> bool:
    """True for on-policy loops, including replay variants like on_policy+replay."""
    return training_mode == "on_policy" or training_mode.startswith("on_policy+")


@dataclass
class TrainingResumeState:
    """Loop counters and tracker state restored when resuming a run."""

    global_step: int = 0
    start_epoch: int = 0
    start_step: int = 0
    start_resample_round: int = 0
    start_update_cycle: int = 0
    training_mode: str = "on_policy"
    checkpoint_path: str | None = None
    outcome_counts: dict[str, int] | None = None
    topology_counts: dict[str, int] | None = None
    grpo_trainer_state: dict | None = None

    @property
    def seen_outcomes(self) -> set[str]:
        if not self.outcome_counts:
            return set()
        return set(self.outcome_counts)


def _atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_metrics_rows(path: Path, rows: list[dict]) -> None:
    text = "".join(json.dumps(row) + "\n" for row in rows)
    _atomic_write_text(path, text)


def load_metrics_rows(
    metrics_path: str | Path,
    *,
    repair_trailing_record: bool = False,
) -> list[dict]:
    path = Path(metrics_path)
    if not path.exists():
        return []
    lines = path.read_text().splitlines()
    rows = []
    for index, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            has_later_record = any(candidate.strip() for candidate in lines[index + 1 :])
            if has_later_record or not repair_trailing_record:
                raise
            print(f"warning: removing partial trailing metrics record from {path}")
            _write_metrics_rows(path, rows)
            break
    return rows


def resolve_checkpoint_path(run_dir: str | Path, checkpoint_name: str | None = None) -> Path:
    root = Path(run_dir)
    if checkpoint_name:
        path = root / checkpoint_name
        if not path.exists():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path

    final_path = root / "final_checkpoint.pt"
    if final_path.exists():
        return final_path

    epoch_ckpts = sorted(root.glob("checkpoint_epoch*.pt"))
    if epoch_ckpts:
        return epoch_ckpts[-1]

    round_ckpts = sorted(root.glob("checkpoint_round*.pt"))
    if round_ckpts:
        return round_ckpts[-1]

    legacy = root / "generator_checkpoint.pt"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(f"no checkpoint found in {root}")


def tracker_from_counts(
    outcome_counts: dict[str, int] | None,
    topology_counts: dict[str, int] | None,
) -> OutcomeTracker:
    tracker = OutcomeTracker()
    if outcome_counts:
        tracker.outcome_counts.update(outcome_counts)
    if topology_counts:
        tracker.topology_counts.update(topology_counts)
    tracker.total = sum(tracker.outcome_counts.values()) or sum(tracker.topology_counts.values())
    return tracker


def save_training_state(
    output_dir: str | Path,
    state: TrainingResumeState,
    *,
    grpo_trainer=None,
) -> None:
    payload = asdict(state)
    payload.pop("grpo_trainer_state", None)
    path = Path(output_dir) / TRAINING_STATE_FILE

    if grpo_trainer is not None and hasattr(grpo_trainer, "state_dict"):
        trainer_path = Path(output_dir) / TRAINER_STATE_FILE
        trainer_temporary = trainer_path.with_name(
            f".{trainer_path.name}.{os.getpid()}.tmp"
        )
        torch.save(grpo_trainer.state_dict(), trainer_temporary)
        os.replace(trainer_temporary, trainer_path)

    _atomic_write_text(path, json.dumps(payload, indent=2))


def load_training_state(output_dir: str | Path) -> TrainingResumeState | None:
    path = Path(output_dir) / TRAINING_STATE_FILE
    if not path.exists():
        return None
    with path.open() as handle:
        raw = json.load(handle)
    return TrainingResumeState(
        global_step=int(raw.get("global_step", 0)),
        start_epoch=int(raw.get("start_epoch", raw.get("epoch", 0))),
        start_step=int(raw.get("start_step", 0)),
        start_resample_round=int(raw.get("start_resample_round", raw.get("resample_round", 0))),
        start_update_cycle=int(raw.get("start_update_cycle", raw.get("update_cycle", 0))),
        training_mode=str(raw.get("training_mode", "on_policy")),
        checkpoint_path=raw.get("checkpoint_path"),
        outcome_counts=raw.get("outcome_counts"),
        topology_counts=raw.get("topology_counts"),
        grpo_trainer_state=raw.get("grpo_trainer_state"),
    )


def _infer_on_policy_start(rows: list[dict], steps_per_epoch: int) -> tuple[int, int, int]:
    last = rows[-1]
    global_step = int(last["global_step"]) + 1
    epoch = int(last["epoch"])
    step = int(last["step"])
    if step + 1 >= steps_per_epoch:
        return global_step, epoch + 1, 0
    return global_step, epoch, step + 1


def _infer_policy_is_start(rows: list[dict], update_cycles: int) -> tuple[int, int, int]:
    last = rows[-1]
    global_step = int(last["global_step"]) + 1
    resample_round = int(last["resample_round"])
    update_cycle = int(last["update_cycle"])
    if update_cycle + 1 >= update_cycles:
        return global_step, resample_round + 1, 0
    # Mid-round resume is unsupported (behavior buffer not persisted).
    return global_step, resample_round + 1, 0


def infer_resume_state_from_metrics(
    rows: list[dict],
    *,
    training_mode: str,
    steps_per_epoch: int,
    update_cycles: int,
) -> TrainingResumeState:
    if not rows:
        return TrainingResumeState(training_mode=training_mode)

    tracker = OutcomeTracker()
    seen: set[str] = set()
    for row in rows:
        # Reconstruct approximate tracker from per-step batch stats when counts absent.
        # Exact counts require training_state.json; this is a fallback only.
        pass

    last = rows[-1]
    outcome_counts = None
    topology_counts = None

    if training_mode == "policy_is":
        global_step, start_round, start_cycle = _infer_policy_is_start(rows, update_cycles)
        if int(last.get("update_cycle", 0)) + 1 < update_cycles:
            print(
                "warning: policy-IS resume from metrics skips incomplete resample round "
                "(behavior buffer is not checkpointed)"
            )
        return TrainingResumeState(
            global_step=global_step,
            start_resample_round=start_round,
            start_update_cycle=start_cycle,
            training_mode=training_mode,
            outcome_counts=outcome_counts,
            topology_counts=topology_counts,
        )

    global_step, start_epoch, start_step = _infer_on_policy_start(rows, steps_per_epoch)
    return TrainingResumeState(
        global_step=global_step,
        start_epoch=start_epoch,
        start_step=start_step,
        training_mode=training_mode,
        outcome_counts=outcome_counts,
        topology_counts=topology_counts,
    )


def load_epoch_summaries(output_dir: str | Path) -> list[dict]:
    path = Path(output_dir) / "epoch_summaries.json"
    if not path.exists():
        return []
    with path.open() as handle:
        data = json.load(handle)
    return data if isinstance(data, list) else []


def _metrics_committed_by_checkpoint(
    rows: list[dict],
    resume: TrainingResumeState,
) -> list[dict]:
    if is_on_policy_training_mode(resume.training_mode):
        return [
            row
            for row in rows
            if int(row.get("epoch", -1)) < resume.start_epoch
            or (
                int(row.get("epoch", -1)) == resume.start_epoch
                and int(row.get("step", -1)) < resume.start_step
            )
        ]
    return [
        row
        for row in rows
        if int(row.get("resample_round", -1)) < resume.start_resample_round
        or (
            int(row.get("resample_round", -1)) == resume.start_resample_round
            and int(row.get("update_cycle", -1)) < resume.start_update_cycle
        )
    ]


def prepare_resume(
    run_dir: str | Path,
    *,
    checkpoint_name: str | None,
    training_mode: str,
    steps_per_epoch: int,
    update_cycles: int,
    target_epochs: int,
    target_resample_rounds: int,
) -> tuple[TrainingResumeState, Path]:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"resume run directory not found: {root}")

    metrics_path = root / "metrics.jsonl"
    saved = load_training_state(root)
    effective_checkpoint_name = checkpoint_name
    if (
        effective_checkpoint_name is None
        and saved is not None
        and saved.checkpoint_path
    ):
        committed_checkpoint = root / Path(saved.checkpoint_path).name
        if committed_checkpoint.exists():
            effective_checkpoint_name = committed_checkpoint.name
    checkpoint_path = resolve_checkpoint_path(root, effective_checkpoint_name)
    rows = load_metrics_rows(metrics_path, repair_trailing_record=True)

    if saved is not None:
        resume = saved
        resume.checkpoint_path = str(checkpoint_path)
        committed_rows = _metrics_committed_by_checkpoint(rows, resume)
        if len(committed_rows) != len(rows):
            print(
                f"warning: removing {len(rows) - len(committed_rows)} metrics records "
                f"newer than committed checkpoint {checkpoint_path.name}"
            )
            _write_metrics_rows(metrics_path, committed_rows)
            rows = committed_rows
        # Advance start pointers from saved "last completed" counters.
        if is_on_policy_training_mode(resume.training_mode):
            if resume.start_step > 0 and resume.start_step < steps_per_epoch:
                pass
            elif resume.start_epoch >= target_epochs:
                raise ValueError(
                    f"run already reached target epochs ({target_epochs}); "
                    f"increase --epochs to continue"
                )
        else:
            if resume.start_resample_round >= target_resample_rounds:
                raise ValueError(
                    f"run already reached target resample rounds ({target_resample_rounds}); "
                    f"increase --epochs/--resample-rounds to continue"
                )
    else:
        if not rows:
            raise ValueError(
                f"cannot infer resume state from empty metrics in {root}; "
                f"need metrics.jsonl or {TRAINING_STATE_FILE}"
            )
        resume = infer_resume_state_from_metrics(
            rows,
            training_mode=training_mode,
            steps_per_epoch=steps_per_epoch,
            update_cycles=update_cycles,
        )
        resume.checkpoint_path = str(checkpoint_path)
        print(f"warning: {TRAINING_STATE_FILE} not found; inferred resume from metrics.jsonl")

    if is_on_policy_training_mode(resume.training_mode) and resume.start_epoch >= target_epochs:
        raise ValueError(
            f"run already completed {resume.start_epoch} epochs (target={target_epochs}); "
            f"set --epochs higher than {resume.start_epoch}"
        )
    if resume.training_mode == "policy_is" and resume.start_resample_round >= target_resample_rounds:
        raise ValueError(
            f"run already completed {resume.start_resample_round} resample rounds "
            f"(target={target_resample_rounds}); increase --epochs or --resample-rounds"
        )

    print(
        f"resuming from {checkpoint_path.name}: "
        f"global_step={resume.global_step} "
        f"mode={resume.training_mode}"
    )
    if is_on_policy_training_mode(resume.training_mode):
        print(f"  continue epochs [{resume.start_epoch}, {target_epochs})")
        if resume.start_step > 0:
            print(f"  mid-epoch resume at step={resume.start_step}")
    else:
        print(f"  continue resample rounds [{resume.start_resample_round}, {target_resample_rounds})")

    return resume, checkpoint_path


def restore_tracker(resume: TrainingResumeState, metrics_rows: list[dict]) -> OutcomeTracker:
    if resume.outcome_counts or resume.topology_counts:
        return tracker_from_counts(resume.outcome_counts, resume.topology_counts)

    if metrics_rows:
        last = metrics_rows[-1]
        print(
            "warning: outcome tracker counts not in training_state.json; "
            f"global diversity counters restart (last cumulative_unique_outcomes="
            f"{last.get('cumulative_unique_outcomes', '?')})"
        )
    return OutcomeTracker()


def make_training_state(
    *,
    global_step: int,
    training_mode: str,
    epoch: int | None = None,
    step: int | None = None,
    steps_per_epoch: int | None = None,
    resample_round: int | None = None,
    update_cycle: int | None = None,
    update_cycles: int | None = None,
    checkpoint_path: str | None = None,
    tracker: OutcomeTracker | None = None,
) -> TrainingResumeState:
    """Build resume pointers for the *next* training step."""
    state = TrainingResumeState(
        global_step=global_step,
        training_mode=training_mode,
        checkpoint_path=checkpoint_path,
    )
    if tracker is not None:
        state.outcome_counts = dict(tracker.outcome_counts)
        state.topology_counts = dict(tracker.topology_counts)

    if is_on_policy_training_mode(training_mode):
        assert epoch is not None and step is not None and steps_per_epoch is not None
        if step + 1 >= steps_per_epoch:
            state.start_epoch = epoch + 1
            state.start_step = 0
        else:
            state.start_epoch = epoch
            state.start_step = step + 1
    else:
        assert resample_round is not None and update_cycle is not None and update_cycles is not None
        if update_cycle + 1 >= update_cycles:
            state.start_resample_round = resample_round + 1
            state.start_update_cycle = 0
        else:
            state.start_resample_round = resample_round
            state.start_update_cycle = update_cycle + 1
    return state


def load_generator_checkpoint(generator, checkpoint_path: str | Path) -> None:
    generator.load(str(checkpoint_path))


def maybe_load_trainer_state(trainer, resume: TrainingResumeState, output_dir: str | Path) -> None:
    if trainer is None:
        return
    trainer_path = Path(output_dir) / TRAINER_STATE_FILE
    if trainer_path.exists():
        trainer.load_state_dict(torch.load(trainer_path, map_location="cpu"))
        print("  restored GRPO/IPS optimizer state")
        return
    if resume.grpo_trainer_state:
        trainer.load_state_dict(resume.grpo_trainer_state)
        print("  restored GRPO/IPS optimizer state (legacy)")


def resolve_output_dir(exp_cfg) -> str:
    if getattr(exp_cfg, "resume_from", None):
        return os.path.abspath(exp_cfg.resume_from)
    from grpo_experiments.utils import build_output_dir

    method = getattr(exp_cfg, "method", "grpo")
    if callable(method):
        method = method()
    return build_output_dir(exp_cfg.output_root, method, exp_cfg.run_name)
