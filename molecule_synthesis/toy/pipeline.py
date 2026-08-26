"""Train all objectives on a tiny multi-route molecule synthesis DAG.

No RGFN, RDKit, DGL, or chemistry data are required; only PyTorch is used.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from molecule_synthesis.methods import METHODS, normalize_method_name
from molecule_synthesis.objective_math import compute_grpo_policy_loss


@dataclass(frozen=True)
class ToyReactionDAG:
    molecule_names: tuple[str, ...] = ("Mol-A", "Mol-B", "Mol-C", "Mol-D")
    rewards: tuple[float, ...] = (1.0, 2.0, 4.0, 8.0)
    route_counts: tuple[int, ...] = (1, 2, 4, 8)

    def tensors(self, device: torch.device) -> tuple[torch.Tensor, ...]:
        reward = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        counts = torch.tensor(self.route_counts, dtype=torch.long, device=device)
        path_to_molecule = torch.repeat_interleave(
            torch.arange(len(self.rewards), device=device), counts
        )
        # The fixed backward policy chooses uniformly among valid synthesis
        # routes for the terminal molecule.
        log_pb = -torch.log(counts.float())[path_to_molecule]
        return reward, counts, path_to_molecule, log_pb

    def reward_target(self, device: torch.device) -> torch.Tensor:
        reward = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        return reward / reward.sum()

    def multiplicity_target(self, device: torch.device) -> torch.Tensor:
        reward = torch.tensor(self.rewards, dtype=torch.float32, device=device)
        counts = torch.tensor(self.route_counts, dtype=torch.float32, device=device)
        target = reward * counts
        return target / target.sum()


def _terminal_distribution(
    logits: torch.Tensor, path_to_molecule: torch.Tensor, n_molecules: int
) -> torch.Tensor:
    path_probability = torch.softmax(logits, dim=0)
    terminal_probability = torch.zeros(
        n_molecules, dtype=path_probability.dtype, device=path_probability.device
    )
    return terminal_probability.scatter_add(0, path_to_molecule, path_probability)


def _reverse_log_probs(
    reverse_logits: torch.Tensor, path_to_molecule: torch.Tensor, n_molecules: int
) -> torch.Tensor:
    """Normalize learned reverse logits separately for each terminal molecule."""
    output = torch.empty_like(reverse_logits)
    for molecule in range(n_molecules):
        mask = path_to_molecule == molecule
        output[mask] = torch.log_softmax(reverse_logits[mask], dim=0)
    return output


def train_method(
    method: str,
    *,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    device: torch.device,
) -> dict:
    method = normalize_method_name(method)
    dag = ToyReactionDAG()
    rewards, route_counts, path_to_molecule, _uniform_log_pb = dag.tensors(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    # A non-uniform route initialization makes the reverse learner do real work:
    # q_phi must track P_F(route | molecule), not merely stay uniform.
    initial_logits = torch.cat(
        [
            torch.linspace(-0.8, 0.8, int(count.item()), device=device)
            for count in route_counts
        ]
    )
    logits = torch.nn.Parameter(initial_logits)
    reverse_logits = torch.nn.Parameter(torch.zeros(len(path_to_molecule), device=device))
    log_z = torch.nn.Parameter(torch.tensor(0.0, device=device))
    if method == "rgfn":
        optimizer = torch.optim.Adam(
            [logits, reverse_logits, log_z], lr=learning_rate
        )
        reverse_optimizer = None
    else:
        optimizer = torch.optim.Adam([logits], lr=learning_rate)
        reverse_optimizer = (
            torch.optim.Adam([reverse_logits], lr=0.05)
            if method == "mips_grpo"
            else None
        )
    history = []

    for step in range(steps):
        path_log_probs = torch.log_softmax(logits, dim=0)
        reverse_log_probs = _reverse_log_probs(
            reverse_logits, path_to_molecule, len(dag.molecule_names)
        )
        sampled_paths = torch.multinomial(
            path_log_probs.exp(), batch_size, replacement=True, generator=generator
        )
        sampled_log_pf = path_log_probs[sampled_paths]
        sampled_molecules = path_to_molecule[sampled_paths]
        sampled_rewards = rewards[sampled_molecules]

        if method == "rgfn":
            residual = (
                log_z
                + sampled_log_pf
                - reverse_log_probs[sampled_paths]
                - sampled_rewards.log()
            )
            loss = residual.square().mean()
            ess_fraction = 1.0
        else:
            if method == "grpo":
                scaled_rewards = sampled_rewards
                ess_fraction = 1.0
            elif method == "count_ips_grpo":
                counts = torch.bincount(
                    sampled_molecules, minlength=len(dag.molecule_names)
                ).float()
                empirical_probability = counts[sampled_molecules] / batch_size
                scaled_rewards = sampled_rewards / empirical_probability.clamp_min(1e-6)
                normalized = scaled_rewards / scaled_rewards.sum().clamp_min(1e-8)
                ess_fraction = float(
                    (1.0 / normalized.square().sum().clamp_min(1e-8) / batch_size).item()
                )
            else:
                log_weight = (
                    sampled_rewards.log()
                    + reverse_log_probs[sampled_paths].detach()
                    - sampled_log_pf.detach()
                )
                scaled_rewards = torch.exp(log_weight - log_weight.max())
                ess = scaled_rewards.sum().square() / scaled_rewards.square().sum().clamp_min(
                    1e-8
                )
                ess_fraction = float((ess / batch_size).item())

            advantages = (scaled_rewards - scaled_rewards.mean()) / scaled_rewards.std(
                unbiased=False
            ).clamp_min(1e-8)
            loss, _ = compute_grpo_policy_loss(
                sampled_log_pf[:, None],
                advantages,
                log_paths_pf_old=sampled_log_pf.detach()[:, None],
                clip_eps=0.2,
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Learned-reverse MIPS follows the phylogenetics update ordering: form
        # forward advantages with the old q_phi, update P_F, then fit q_phi by
        # maximum likelihood on the sampled synthesis trajectories.
        reverse_loss_value = 0.0
        if method == "mips_grpo":
            updated_reverse_log_probs = _reverse_log_probs(
                reverse_logits, path_to_molecule, len(dag.molecule_names)
            )
            reverse_loss = -updated_reverse_log_probs[sampled_paths].mean()
            reverse_optimizer.zero_grad()
            reverse_loss.backward()
            reverse_optimizer.step()
            reverse_loss_value = float(reverse_loss.item())

        if step == 0 or (step + 1) % 25 == 0 or step == steps - 1:
            with torch.no_grad():
                terminal = _terminal_distribution(
                    logits, path_to_molecule, len(dag.molecule_names)
                )
                history.append(
                    {
                        "step": step + 1,
                        "loss": float(loss.item()),
                        "l1_to_reward_target": float(
                            (terminal - dag.reward_target(device)).abs().sum().item()
                        ),
                        "ess_fraction": float(ess_fraction),
                        "reverse_loss": reverse_loss_value,
                    }
                )

    with torch.no_grad():
        terminal = _terminal_distribution(logits, path_to_molecule, len(dag.molecule_names))
        reward_target = dag.reward_target(device)
        multiplicity_target = dag.multiplicity_target(device)
        path_probability = torch.softmax(logits, dim=0)
        conditional_log_pf = torch.log(path_probability) - torch.log(
            terminal[path_to_molecule]
        )
        learned_log_pb = _reverse_log_probs(
            reverse_logits, path_to_molecule, len(dag.molecule_names)
        )
        reverse_kl = torch.sum(
            path_probability * (conditional_log_pf - learned_log_pb)
        )
    return {
        "method": method,
        "label": METHODS[method].label,
        "molecule_names": list(dag.molecule_names),
        "rewards": list(dag.rewards),
        "route_counts": list(dag.route_counts),
        "learned_distribution": terminal.cpu().tolist(),
        "reward_target": reward_target.cpu().tolist(),
        "multiplicity_target": multiplicity_target.cpu().tolist(),
        "l1_to_reward_target": float((terminal - reward_target).abs().sum().item()),
        "l1_to_multiplicity_target": float(
            (terminal - multiplicity_target).abs().sum().item()
        ),
        "top_molecule_probability": float(terminal[-1].item()),
        "reverse_conditional_kl": float(reverse_kl.item()),
        "history": history,
    }


def _assert_expected(results: dict[str, dict]) -> None:
    failures = []
    for method in ("rgfn", "count_ips_grpo", "mips_grpo"):
        if method in results and results[method]["l1_to_reward_target"] >= 0.05:
            failures.append(f"{method} did not recover the reward target")
    if "grpo" in results and results["grpo"]["top_molecule_probability"] <= 0.90:
        failures.append("plain GRPO did not concentrate on the maximum-reward molecule")
    if "mips_grpo" in results and results["mips_grpo"]["reverse_conditional_kl"] >= 0.02:
        failures.append("MIPS learned reverse did not fit P_F(route|molecule)")
    if failures:
        raise AssertionError("; ".join(failures))


def _write_results(output_dir: Path, results: dict[str, dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with (output_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method",
            "l1_to_reward_target",
            "l1_to_multiplicity_target",
            "top_molecule_probability",
            "reverse_conditional_kl",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results.values():
            writer.writerow({key: result[key] for key in fieldnames})


def _print_table(results: dict[str, dict]) -> None:
    print("method                  L1(reward)   P(Mol-D)   reverse-KL")
    print("----------------------  -----------  --------   ----------")
    for method, result in results.items():
        print(
            f"{method:<22}  {result['l1_to_reward_target']:>11.4f}  "
            f"{result['top_molecule_probability']:>8.4f}   "
            f"{result['reverse_conditional_kl']:>10.5f}"
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        default="all",
        help="all, rgfn, grpo, count_ips_grpo, or mips_grpo",
    )
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output",
        default="molecule_synthesis/toy/runs/latest",
    )
    parser.add_argument("--assert-expected", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps <= 0 or args.batch_size < 2:
        raise ValueError("steps must be positive and batch-size must be at least 2")
    torch.set_num_threads(1)
    device = torch.device(args.device)
    methods = tuple(METHODS) if args.method == "all" else (normalize_method_name(args.method),)
    results = {
        method: train_method(
            method,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            seed=args.seed,
            device=device,
        )
        for method in methods
    }
    _write_results(Path(args.output).expanduser().resolve(), results)
    _print_table(results)
    if args.assert_expected:
        _assert_expected(results)
        print("toy_verification=PASS")
    print(f"results={Path(args.output).expanduser().resolve() / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
