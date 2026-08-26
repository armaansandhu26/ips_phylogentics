from __future__ import annotations

from pathlib import Path

from final.configs import SuiteConfig
from final.methods.base import CommandSpec, python_module, repo_script, shared_ppo_train_args
from final.paths import REPO_ROOT


class LearnedReverseRunner:
    name = "learned_reverse"

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
        method_cfg = suite.method_cfg(self.name)
        argv = python_module(
            "learned_reverse_ips.train",
            "--run-name",
            run_name,
            "--cfg",
            str(cfg.relative_to(REPO_ROOT)),
            *shared_ppo_train_args(suite, output_root),
            "--reward-target",
            str(method_cfg.get("reward_target", "shifted_linear")),
            "--reverse-lr",
            str(method_cfg.get("reverse_lr", 1e-3)),
            "--reverse-train-epochs",
            str(method_cfg.get("reverse_train_epochs", 4)),
            "--reverse-hidden-size",
            str(method_cfg.get("reverse_hidden_size", 128)),
            "--reverse-num-layers",
            str(method_cfg.get("reverse_num_layers", 2)),
            "--skip-post-train",
        )
        reverse_policy = method_cfg.get("reverse_policy", "mlp")
        argv += ["--reverse-policy", str(reverse_policy)]
        if resume_from is not None:
            argv += ["--resume-from", str(resume_from)]
        if resume_checkpoint is not None:
            argv += ["--resume-checkpoint", resume_checkpoint]
        rollout_chunk = method_cfg.get("rollout_chunk_size")
        if rollout_chunk is not None:
            # Replace default rollout-chunk-size from shared_ppo_train_args (last batch-size entry).
            idx = argv.index("--rollout-chunk-size")
            argv[idx + 1] = str(rollout_chunk)
        mini_batch_splits = method_cfg.get("mini_batch_splits")
        if mini_batch_splits is not None:
            argv += ["--mini-batch-splits", str(mini_batch_splits)]
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
            "grpo_experiments/scripts/sample_learned_reverse_full_diagnostics.py",
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
            "grpo_experiments/scripts/plot_full_checkpoint_vs_reward_reference.py",
            "--samples",
            str(samples_path),
            "--output-dir",
            str(plot_dir),
            "--plot-method",
            self.plot_method(),
            "--shared-reference",
        )
        return CommandSpec(argv=argv, cwd=REPO_ROOT)

    def build_training_curves_command(self, run_dir: Path) -> CommandSpec | None:
        argv = python_module(
            "learned_reverse_ips.post_train",
            "--run-dir",
            str(run_dir),
            "--skip-sampling",
            "--skip-sampling-plots",
        )
        return CommandSpec(argv=argv, cwd=REPO_ROOT)

    def plot_method(self) -> str:
        return "learned-reverse"

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
