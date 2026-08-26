from __future__ import annotations

from pathlib import Path

from final.configs import SuiteConfig
from final.methods.base import CommandSpec, python_module, repo_script, shared_ppo_train_args
from final.paths import REPO_ROOT


class GrpoRunner:
    name = "grpo"

    def output_root(self, suite: SuiteConfig) -> Path:
        return suite.method_output_root(self.name)

    def build_train_command(
        self,
        suite: SuiteConfig,
        *,
        output_root: Path,
        run_name: str,
        resume_from: Path | None = None,
        resume_checkpoint: str | None = None,
    ) -> CommandSpec:
        cfg = suite.resolve_cfg_path(self.name)
        argv = python_module(
            "grpo_experiments.train",
            "--method",
            "grpo",
            "--run-name",
            run_name,
            "--cfg",
            str(cfg.relative_to(REPO_ROOT)),
            *shared_ppo_train_args(suite, output_root),
        )
        if resume_from is not None:
            argv += ["--resume-from", str(resume_from)]
        if resume_checkpoint is not None:
            argv += ["--resume-checkpoint", resume_checkpoint]
        return CommandSpec(argv=argv, cwd=REPO_ROOT)

    def build_sample_command(
        self,
        suite: SuiteConfig,
        run_dir: Path,
        *,
        num_trees: int,
        batch_size: int,
        device: str,
    ) -> CommandSpec:
        samples = run_dir / f"sampled_full_diagnostics_{num_trees}.npz"
        argv = repo_script(
            "grpo_experiments/scripts/sample_ppo_full_diagnostics.py",
            "--checkpoint",
            str(run_dir),
            "-n",
            str(num_trees),
            "--batch-size",
            str(batch_size),
            "--seed",
            "0",
            "--device",
            device,
            "--reward-shift",
            str(suite.log_score_shift),
            "--output",
            str(samples),
        )
        return CommandSpec(argv=argv, cwd=REPO_ROOT)

    def build_plot_command(
        self,
        run_dir: Path,
        samples_path: Path,
    ) -> CommandSpec:
        plot_dir = run_dir / "plots" / "mlp_shifted_linear_reference_1000k"
        argv = repo_script(
            "grpo_experiments/scripts/plot_empirical_signature_vs_reward.py",
            "--samples",
            str(samples_path),
            "--output",
            str(plot_dir / "model_probability_vs_reward.png"),
            "--title",
            "Plain GRPO",
        )
        return CommandSpec(argv=argv, cwd=REPO_ROOT)

    def build_training_curves_command(self, run_dir: Path) -> CommandSpec | None:
        plot_dir = run_dir / "plots"
        return CommandSpec(
            argv=repo_script(
                "grpo_experiments/scripts/evaluate_runs.py",
                "--run-dirs",
                str(run_dir),
                "--labels",
                "grpo",
                "--output-dir",
                str(plot_dir),
            ),
            cwd=REPO_ROOT,
        )

    def plot_method(self) -> str:
        return "ppo"

    def expected_checkpoint(self, run_dir: Path) -> Path:
        return run_dir / "final_checkpoint.pt"

    def comparison_metrics_path(self, run_dir: Path, num_trees: int) -> Path:
        return (
            run_dir
            / "plots"
            / "mlp_shifted_linear_reference_1000k"
            / "comparison_metrics.json"
        )

    def run_ready_marker(self, run_dir: Path) -> Path | None:
        return run_dir / "resolved_config.yaml"
