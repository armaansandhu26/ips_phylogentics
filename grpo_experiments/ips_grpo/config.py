"""
Configuration for IPS-GRPO runs.

Separate from grpo_experiments/config.py so PhyloGFN / GRPO / IPS-GRPO stay
independent entry points. Shared batch and model settings mirror the other runners.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Optional

from grpo_experiments.ips_grpo.policy_loss_modes import ALL_POLICY_LOSS_MODES, PolicyLossMode

OutcomeLevel = Literal["signature", "topology"]
AdvantageRewardMode = Literal["exp_linear", "log_reward"]
IPSPropensityMode = Literal["count", "exact"]

_CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "configs")
DEFAULT_PRESET_FILE = os.path.join(_CONFIG_DIR, "ips_grpo_presets.json")

# Fields that may appear in a preset JSON object (keys starting with "_" are ignored).
_PRESET_FIELD_ALIASES = {
    "cfg": "cfg_path",
    "dataset": "dataset_path",
    "output": "output_root",
}


@dataclass
class IPSExperimentConfig:
    """Knobs for an IPS-GRPO training run."""

    cfg_path: str = (
        "src/configs/benchmark_dna_cfgs/discrete_branch_lengths/"
        "cfg_0.001binsize_50bins_temperature_anneal_0.4.yaml"
    )
    dataset_path: str = "dataset/benchmark_datasets/DS1_reduced.pickle"
    output_root: str = "grpo_experiments/runs"
    run_name: Optional[str] = None
    seed: int = 0
    device: Optional[str] = None

    epochs: int = 100
    steps_per_epoch: int = 20

    on_policy_batch_size: int = 64
    replay_batch_size: int = 0
    replay_buffer_size: int = 512
    mini_batch_splits: int = 1
    disable_replay: bool = False

    grpo_lr: float = 1e-4
    grpo_max_grad_norm: float = 1.0
    grpo_advantage_eps: float = 1e-8
    grpo_clip_eps: float = 0.2
    grpo_clip_eps_high: float | None = None
    grpo_entropy_coef: float = 0.0
    grpo_num_iterations: int = 1

    ips_prob_floor: float = 1e-6
    """eps in r_tilde = r / max(p_hat(o), eps) — arXiv:2601.21669 Eq. 9."""

    ips_propensity_mode: IPSPropensityMode = "count"
    """count: legacy batch outcome counts; exact: exp(-log p_theta(tau)) with SNIPS."""

    max_inverse_weight: float = 2560.0
    """Cap for exact inverse trajectory propensity before SNIPS normalization (legacy path)."""

    ips_weight_temperature: float = 1.0
    """exact mode: beta in (0, 1] applied to log inverse propensity before SNIPS.

    beta == 1 is pure exact IPS (heavy-tailed, collapses at scale); beta -> 0 recovers
    plain GRPO. Values around 0.2-0.5 keep ESS healthy on the full model. Any value
    other than 1.0 switches to the numerically stable log-space SNIPS path."""

    snips_truncate_ratio: float | None = None
    """exact mode: truncate SNIPS weights (mean 1) to this multiple of the mean, then
    renormalize. e.g. 10.0 bounds any single sample to <=10x average influence. None
    keeps the legacy raw-exp + absolute-cap behavior when combined with temperature 1.0."""

    ips_target_ess_fraction: float | None = None
    """exact mode: if set (e.g. 0.5), auto-solve the temperature beta each batch so the
    SNIPS effective sample size stays at this fraction of the group. Overrides
    ips_weight_temperature and adapts as the policy sharpens. Recommended over a fixed
    temperature; start at 0.5."""

    enable_policy_is: bool = False
    """If True: sample buffer under behavior policy, replay with pi_new/pi_old weights."""

    resample_rounds: Optional[int] = None
    update_cycles: Optional[int] = None
    buffer_size: Optional[int] = None
    rollout_chunk_size: int = 2048

    only_train_tree_model: bool | None = None
    """If set, override GFN.MODEL.ONLY_TRAIN_TREE_MODEL (False = train tree + edges)."""

    edge_rep_grad_alpha: float | None = None
    """Override GFN.MODEL.EDGE_REP_GRAD_ALPHA; None keeps the YAML/default value."""

    outcome_level: OutcomeLevel = "topology"
    print_every: int = 1
    checkpoint_every: int = 0
    dump_advantage_groups: bool = False
    advantage_reward_mode: AdvantageRewardMode = "log_reward"
    policy_loss_mode: PolicyLossMode = "ppo"
    """ppo: IPS-scaled advantages + PPO clip. split_ppo / magnitude_weighted_ppo: tree-edge credit split."""

    tree_loss_weight: float = 0.5
    """split_ppo only: weight on the tree PPO term."""

    edge_loss_weight: float = 0.5
    """split_ppo only: weight on the edge PPO term."""

    tempered_ips_tau: float | None = None
    """Fixed temperature tau for tempered_log_ips. None => tau = std(ell) / tempered_ips_tau_divisor per batch."""

    tempered_ips_tau_divisor: float = 3.0
    """Batch-adaptive tau divisor when tempered_ips_tau is None."""

    log_score_decimals: int | None = None
    """If set, round log_scores (and log_rewards) to this many decimal places everywhere."""

    cpu_threads: int = 0
    """Max CPU threads per process; 0 uses YAML/env/default resolution."""

    resume_from: Optional[str] = None
    resume_checkpoint: Optional[str] = None

    extra: dict = field(default_factory=dict)

    @property
    def method(self) -> str:
        return "ips_grpo"

    @property
    def effective_resample_rounds(self) -> int:
        return self.epochs if self.resample_rounds is None else self.resample_rounds

    @property
    def effective_update_cycles(self) -> int:
        return self.steps_per_epoch if self.update_cycles is None else self.update_cycles

    @property
    def effective_buffer_size(self) -> int:
        return self.on_policy_batch_size if self.buffer_size is None else self.buffer_size

    @property
    def grpo_group_size(self) -> int:
        if self.enable_policy_is:
            return self.effective_buffer_size
        replay = 0 if self.disable_replay else self.replay_batch_size
        return self.on_policy_batch_size + replay

    @property
    def effective_replay_batch_size(self) -> int:
        return 0 if self.disable_replay else self.replay_batch_size

    def to_dict(self) -> dict:
        d = asdict(self)
        d["method"] = self.method
        d["grpo_group_size"] = self.grpo_group_size
        d["effective_replay_batch_size"] = self.effective_replay_batch_size
        return d

    def save_json(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _normalize_preset_entry(raw: dict[str, Any]) -> dict[str, Any]:
    """Map preset JSON keys to IPSExperimentConfig / argparse dest names."""
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key.startswith("_"):
            continue
        dest = _PRESET_FIELD_ALIASES.get(key, key)
        out[dest] = value
    return out


def load_preset_file(path: str) -> dict[str, dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Preset file must be a JSON object: {path}")
    return {
        name: _normalize_preset_entry(entry)
        for name, entry in data.items()
        if not name.startswith("_") and isinstance(entry, dict)
    }


def load_preset(path: str, name: str) -> dict[str, Any]:
    presets = load_preset_file(path)
    if name not in presets:
        known = ", ".join(sorted(presets))
        raise ValueError(f"Unknown preset {name!r}. Available: {known}")
    return presets[name]


def list_preset_names(path: str = DEFAULT_PRESET_FILE) -> list[str]:
    return sorted(load_preset_file(path))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train phylogenetic tree sampling with IPS-GRPO (Sinha et al. 2026).\n\n"
            "Compare against:\n"
            "  python -m grpo_experiments.train --method phylgfn ...\n"
            "  python -m grpo_experiments.train --method grpo ...\n\n"
            "Outcome-level presets (topology vs signature p_hat):\n"
            "  python -m grpo_experiments.ips_grpo.train --list-presets\n"
            "  python -m grpo_experiments.ips_grpo.train --preset topology_sanity\n"
            "  python -m grpo_experiments.ips_grpo.train --preset signature_sanity\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_argument_group("presets (grpo_experiments/configs/ips_grpo_presets.json)")
    g.add_argument(
        "--preset",
        default=None,
        metavar="NAME",
        help="Load matched settings from ips_grpo_presets.json (CLI flags override).",
    )
    g.add_argument(
        "--preset-file",
        default=DEFAULT_PRESET_FILE,
        help="JSON preset file (default: grpo_experiments/configs/ips_grpo_presets.json).",
    )
    g.add_argument(
        "--list-presets",
        action="store_true",
        help="Print preset names and exit.",
    )

    p.add_argument("--cfg", dest="cfg_path", default=IPSExperimentConfig.cfg_path)
    p.add_argument("--dataset", dest="dataset_path", default=IPSExperimentConfig.dataset_path)
    p.add_argument("--output", dest="output_root", default=IPSExperimentConfig.output_root)
    p.add_argument("--run-name", dest="run_name", default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default=None)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=20)

    g = p.add_argument_group("batch / group size G")
    g.add_argument("--on-policy-batch-size", type=int, default=64)
    g.add_argument("--replay-batch-size", type=int, default=0)
    g.add_argument("--replay-buffer-size", type=int, default=512)
    g.add_argument("--mini-batch-splits", type=int, default=1)
    g.add_argument("--disable-replay", action="store_true")

    g = p.add_argument_group("optimizer")
    g.add_argument("--grpo-lr", type=float, default=1e-4)
    g.add_argument("--grpo-max-grad-norm", type=float, default=1.0)
    g.add_argument("--grpo-advantage-eps", type=float, default=1e-8)
    g.add_argument(
        "--grpo-clip-eps",
        type=float,
        default=0.2,
        help="PPO clip epsilon for GRPO surrogate (TRL default 0.2). 0 disables clipping.",
    )
    g.add_argument("--grpo-clip-eps-high", type=float, default=None)
    g.add_argument("--grpo-entropy-coef", type=float, default=0.0)
    g.add_argument(
        "--grpo-num-iterations",
        type=int,
        default=1,
        help="On-policy only: reuse each rollout for this many updates (TRL mu).",
    )

    g = p.add_argument_group("IPS (arXiv:2601.21669)")
    g.add_argument(
        "--ips-prob-floor",
        type=float,
        default=1e-6,
        help="Floor on batch outcome probability before inverting.",
    )
    g.add_argument(
        "--ips-propensity-mode",
        choices=["count", "exact"],
        default="count",
        help="count = legacy count propensity; exact = exp(-log p_theta(tau)) with cap + SNIPS.",
    )
    g.add_argument(
        "--max-inverse-weight",
        type=float,
        default=2560.0,
        help="Cap for exact inverse trajectory propensity before SNIPS normalization (legacy path).",
    )
    g.add_argument(
        "--ips-weight-temperature",
        type=float,
        default=1.0,
        help=(
            "exact mode: beta in (0,1] on the log inverse propensity before SNIPS. "
            "1.0 = pure exact IPS (collapses at scale); ~0.2-0.5 keeps ESS healthy; "
            "-> 0 = plain GRPO. Any value != 1.0 uses stable log-space SNIPS."
        ),
    )
    g.add_argument(
        "--snips-truncate-ratio",
        type=float,
        default=None,
        help=(
            "exact mode: clip SNIPS weights (mean 1) to this multiple of the mean and "
            "renormalize (truncated IS). e.g. 10 bounds single-sample influence."
        ),
    )
    g.add_argument(
        "--ips-target-ess-fraction",
        type=float,
        default=None,
        help=(
            "exact mode: auto-solve temperature beta each batch to hold SNIPS ESS at "
            "this fraction of the group (e.g. 0.5). Overrides --ips-weight-temperature."
        ),
    )
    g.add_argument(
        "--outcome-level",
        choices=["signature", "topology"],
        default="topology",
        help=(
            "Outcome o for p_hat: 'topology' = tree_topology_id (shape only); "
            "'signature' = topology_id + log_score (3 dp). Use --preset for paired runs."
        ),
    )

    g = p.add_argument_group("policy importance sampling (pi_new / pi_old)")
    g.add_argument(
        "--enable-policy-is",
        action="store_true",
        help=(
            "Sample a fixed buffer per round, replay trajectories with pi_new/pi_old. "
            "Uses --epochs/--steps-per-epoch/--on-policy-batch-size as defaults when "
            "round/cycle/buffer overrides are omitted."
        ),
    )
    g.add_argument("--resample-rounds", type=int, default=None)
    g.add_argument("--update-cycles", type=int, default=None)
    g.add_argument("--buffer-size", type=int, default=None)
    g.add_argument("--rollout-chunk-size", type=int, default=2048)

    g = p.add_argument_group("model ablations")
    g.add_argument(
        "--full-model",
        action="store_true",
        help=(
            "Train tree topology and categorical edge lengths "
            "(ONLY_TRAIN_TREE_MODEL=false). Default follows YAML (tree-only)."
        ),
    )
    g.add_argument(
        "--tree-only",
        action="store_true",
        help="Train topology only with fixed edge lengths (ONLY_TRAIN_TREE_MODEL=true).",
    )
    g.add_argument(
        "--edge-rep-grad-alpha",
        type=float,
        default=None,
        help=(
            "Scale edge-loss gradients flowing into tree representations. "
            "0 detaches edge inputs; 1 keeps the original coupled path."
        ),
    )

    g = p.add_argument_group("logging")
    g.add_argument("--print-every", type=int, default=1)
    g.add_argument("--checkpoint-every", type=int, default=0)
    g.add_argument(
        "--dump-advantage-groups",
        action="store_true",
        help="Save per-group advantage vectors and shape stats to advantage_groups/.",
    )
    g.add_argument(
        "--advantage-reward-mode",
        choices=["exp_linear", "log_reward"],
        default="log_reward",
        help="exp_linear: r=exp(log_r-max); log_reward: r=log_r before group normalization.",
    )
    g.add_argument(
        "--policy-loss-mode",
        choices=list(ALL_POLICY_LOSS_MODES),
        default="ppo",
        help=(
            "Objective function for training. "
            "ppo: IPS-scaled group advantages + PPO surrogate (core/loss.py). "
            "split_ppo: separate tree/edge PPO surrogates (core/loss_split_ppo.py). "
            "magnitude_weighted_ppo: |log p|-weighted combo before PPO (core/loss_magnitude_weighted_ppo.py). "
            "tempered_log_ips: tempered log-space IPS advantages + PPO surrogate. "
            "log_ips: token log(pi_new/pi_old) + log(score) - log(p_hat). "
            "terminal_seq_pf / terminal_token_ratio / terminal_seq_ratio: IPS terminal ablations."
        ),
    )
    g.add_argument(
        "--tree-loss-weight",
        type=float,
        default=0.5,
        help="split_ppo only: weight on tree PPO term (edge weight is --edge-loss-weight).",
    )
    g.add_argument(
        "--edge-loss-weight",
        type=float,
        default=0.5,
        help="split_ppo only: weight on edge PPO term.",
    )
    g.add_argument(
        "--tempered-ips-tau",
        type=float,
        default=None,
        metavar="TAU",
        help=(
            "tempered_log_ips only: fixed batch temperature tau. "
            "Omit for adaptive tau = std(ell) / --tempered-ips-tau-divisor."
        ),
    )
    g.add_argument(
        "--tempered-ips-tau-divisor",
        type=float,
        default=3.0,
        help="tempered_log_ips only: divisor for adaptive tau when --tempered-ips-tau is omitted.",
    )
    g.add_argument(
        "--log-score-decimals",
        type=int,
        default=None,
        help=(
            "Round log_scores to this many decimal places for training, logging, and "
            "signature outcomes. Use 3 to match signature discretization."
        ),
    )
    g.add_argument(
        "--cpu-threads",
        type=int,
        default=0,
        help=(
            "Cap PyTorch/BLAS CPU threads for this process. "
            "0 uses YAML/env/default resolution."
        ),
    )

    g = p.add_argument_group("resume")
    g.add_argument(
        "--resume-from",
        default=None,
        help="Continue training in an existing run directory (same output folder).",
    )
    g.add_argument(
        "--resume-checkpoint",
        default=None,
        help="Checkpoint filename inside --resume-from (default: final or latest epoch/round).",
    )
    return p


def config_from_args(args: argparse.Namespace) -> IPSExperimentConfig:
    if args.full_model and args.tree_only:
        raise ValueError("Use only one of --full-model or --tree-only")
    if args.full_model:
        only_train_tree_model = False
    elif args.tree_only:
        only_train_tree_model = True
    else:
        only_train_tree_model = None

    return IPSExperimentConfig(
        cfg_path=args.cfg_path,
        dataset_path=args.dataset_path,
        output_root=args.output_root,
        run_name=args.run_name,
        seed=args.seed,
        device=args.device,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        on_policy_batch_size=args.on_policy_batch_size,
        replay_batch_size=args.replay_batch_size,
        replay_buffer_size=args.replay_buffer_size,
        mini_batch_splits=args.mini_batch_splits,
        disable_replay=args.disable_replay,
        grpo_lr=args.grpo_lr,
        grpo_max_grad_norm=args.grpo_max_grad_norm,
        grpo_advantage_eps=args.grpo_advantage_eps,
        grpo_clip_eps=args.grpo_clip_eps,
        grpo_clip_eps_high=args.grpo_clip_eps_high,
        grpo_entropy_coef=args.grpo_entropy_coef,
        grpo_num_iterations=args.grpo_num_iterations,
        ips_prob_floor=args.ips_prob_floor,
        ips_propensity_mode=args.ips_propensity_mode,
        max_inverse_weight=args.max_inverse_weight,
        ips_weight_temperature=args.ips_weight_temperature,
        snips_truncate_ratio=args.snips_truncate_ratio,
        ips_target_ess_fraction=args.ips_target_ess_fraction,
        enable_policy_is=args.enable_policy_is,
        resample_rounds=args.resample_rounds,
        update_cycles=args.update_cycles,
        buffer_size=args.buffer_size,
        rollout_chunk_size=args.rollout_chunk_size,
        only_train_tree_model=only_train_tree_model,
        edge_rep_grad_alpha=args.edge_rep_grad_alpha,
        outcome_level=args.outcome_level,
        print_every=args.print_every,
        checkpoint_every=args.checkpoint_every,
        dump_advantage_groups=args.dump_advantage_groups,
        advantage_reward_mode=args.advantage_reward_mode,
        policy_loss_mode=args.policy_loss_mode,
        tree_loss_weight=args.tree_loss_weight,
        edge_loss_weight=args.edge_loss_weight,
        tempered_ips_tau=args.tempered_ips_tau,
        tempered_ips_tau_divisor=args.tempered_ips_tau_divisor,
        log_score_decimals=args.log_score_decimals,
        cpu_threads=args.cpu_threads,
        resume_from=args.resume_from,
        resume_checkpoint=args.resume_checkpoint,
    )


def parse_experiment_config(argv: list[str] | None = None) -> IPSExperimentConfig:
    """Parse CLI args, applying --preset defaults before explicit flag overrides."""
    argv = sys.argv[1:] if argv is None else argv

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default=None)
    pre.add_argument("--preset-file", default=DEFAULT_PRESET_FILE)
    pre.add_argument("--list-presets", action="store_true")
    pre_args, remaining = pre.parse_known_args(argv)

    if pre_args.list_presets:
        for name in list_preset_names(pre_args.preset_file):
            preset = load_preset(pre_args.preset_file, name)
            level = preset.get("outcome_level", "?")
            run_name = preset.get("run_name", "")
            print(f"{name:24s}  outcome_level={level:10s}  run_name={run_name}")
        raise SystemExit(0)

    parser = build_arg_parser()
    if pre_args.preset:
        parser.set_defaults(**load_preset(pre_args.preset_file, pre_args.preset))
    args = parser.parse_args(remaining)
    return config_from_args(args)
