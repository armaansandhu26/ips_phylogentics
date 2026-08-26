from __future__ import annotations

from pathlib import Path

from final.configs import SuiteConfig
from final.methods.base import CommandSpec, repo_script
from final.paths import OG_CODE_ROOT, PAPER_ROOT, REPO_ROOT, PYTHON


class PhylgfnRunner:
    name = "phylgfn"

    def output_root(self, suite: SuiteConfig) -> Path:
        return suite.method_output_root(self.name)

    def _backend(self, suite: SuiteConfig) -> str:
        return str(suite.method_cfg(self.name).get("backend", "og_code"))

    def _backend_root(self, suite: SuiteConfig) -> Path:
        backend = self._backend(suite)
        if backend == "paper":
            return PAPER_ROOT
        if backend == "og_code":
            return OG_CODE_ROOT
        raise ValueError(f"unknown phylgfn backend {backend!r}; use og_code or paper")

    def _dataset_arg(self, suite: SuiteConfig) -> str:
        backend_root = self._backend_root(suite)
        if self._backend(suite) == "paper":
            return "dataset/benchmark_datasets/" + suite.dataset.name
        return "../" + str(suite.dataset.relative_to(REPO_ROOT))

    def _output_arg(self, output_root: Path, run_name: str, backend_root: Path) -> str:
        target = (output_root / run_name).resolve()
        try:
            return str(target.relative_to(backend_root))
        except ValueError:
            return str(target)

    def build_train_command(
        self,
        suite: SuiteConfig,
        *,
        output_root: Path,
        run_name: str,
        resume_from: Path | None = None,
        resume_checkpoint: str | None = None,
    ) -> CommandSpec:
        backend_root = self._backend_root(suite)
        cfg = suite.resolve_cfg_path(self.name)
        cfg_arg = (
            str(cfg.relative_to(backend_root))
            if cfg.is_relative_to(backend_root)
            else str(cfg)
        )
        output_arg = self._output_arg(output_root, run_name, backend_root)
        if resume_from is not None:
            resolved = resume_from.resolve()
            resume_arg = (
                str(resolved.relative_to(backend_root))
                if resolved.is_relative_to(backend_root)
                else str(resolved)
            )
            argv = [
                str(PYTHON),
                "-u",
                "train.py",
                "resume",
                resume_arg,
                self._dataset_arg(suite),
                output_arg,
                "--nb_device",
                "1",
            ]
        else:
            argv = [
                str(PYTHON),
                "-u",
                "train.py",
                cfg_arg,
                self._dataset_arg(suite),
                output_arg,
                "--nb_device",
                "1",
            ]
        return CommandSpec(argv=argv, cwd=backend_root)

    def build_sample_command(
        self,
        suite: SuiteConfig,
        run_dir: Path,
        *,
        num_trees: int,
        batch_size: int,
        device: str,
    ) -> CommandSpec:
        if self._backend(suite) == "paper":
            argv = [
                str(PYTHON),
                "-u",
                "scripts/eval_reward_probability.py",
                "--run-dir",
                str(run_dir),
                "--dataset",
                self._dataset_arg(suite),
                "-n",
                str(num_trees),
                "--batch-size",
                str(batch_size),
                "--device",
                device,
            ]
            return CommandSpec(argv=argv, cwd=PAPER_ROOT)

        argv = repo_script(
            "grpo_experiments/scripts/eval_og_gflownet_reward_probability.py",
            "--run-dir",
            str(run_dir),
            "--dataset",
            str(suite.dataset.relative_to(REPO_ROOT)),
            "-n",
            str(num_trees),
            "--batch-size",
            str(batch_size),
            "--device",
            device,
        )
        return CommandSpec(argv=argv, cwd=REPO_ROOT)

    def build_plot_command(self, run_dir: Path, samples_path: Path) -> CommandSpec | None:
        return None

    def build_training_curves_command(self, run_dir: Path) -> CommandSpec | None:
        return None

    def plot_method(self) -> str:
        return "gflownet"

    def expected_checkpoint(self, run_dir: Path) -> Path:
        checkpoints = sorted((run_dir / "checkpoints").glob("checkpoint_*.pt"))
        if checkpoints:
            return checkpoints[-1]
        return run_dir / "checkpoints" / "checkpoint_latest.pt"

    def comparison_metrics_path(self, run_dir: Path, num_trees: int) -> Path:
        return (
            run_dir
            / "plots"
            / f"reward_probability_eval_{num_trees}"
            / "comparison_metrics.json"
        )

    def run_ready_marker(self, run_dir: Path) -> Path | None:
        return run_dir / "config.yaml"
