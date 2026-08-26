"""Train learned-reverse IPS on the Hyper-Grid toy environment."""

from __future__ import annotations

from final.logging.wandb_logger import WandbSettings
from final.toy.pipeline import _wandb_settings_from_args
from grpo_experiments.hypergrid.config import build_arg_parser, config_from_args
from grpo_experiments.hypergrid.runner import run_experiment


def main() -> None:
    parser = build_arg_parser()
    parser.set_defaults(method="learned_reverse_ips")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="phylogfn-final")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-group", default=None)
    parser.add_argument("--wandb-tags", nargs="*", default=None)
    args = parser.parse_args()
    cfg = config_from_args(args)
    out = run_experiment(cfg, wandb_settings=_wandb_settings_from_args(args))
    print(f"run complete: {out}")


if __name__ == "__main__":
    main()
