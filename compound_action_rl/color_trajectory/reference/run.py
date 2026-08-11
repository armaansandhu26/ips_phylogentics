from __future__ import annotations

import argparse
import importlib.util
import itertools
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from env.grid_environment import GridEnv
from plot.diagnostics import plot_results, plot_sampling_heatmap


AGENT_SPECS: dict[str, tuple[str, str, str]] = {
    "ips-grpo": ("ips-grpo.py", "ips_grpo_agent", "IPSGRPOAgent"),
    "energy-ips": ("energy-ips.py", "energy_ips_agent", "EnergyIPSGRPOAgent"),
    "buf-energy-ips": ("buf_energy-ips.py", "buf_energy_ips_agent", "BufferedEnergyIPSAgent"),
    "grpo": ("grpo.py", "grpo_agent", "GRPOAgent"),
    "flowrl": ("flowrl.py", "flowrl_agent", "FlowRLAgent"),
    "ppo": ("ppo.py", "ppo_agent", "PPOAgent"),
    "mara": ("mara.py", "mara_agent", "MARAAgent"),
}


def load_agent_class(algo: str, use_gpu: bool = False):
    """
    Load agent class from `grid_experiments/agent`.
    """
    algo = str(algo).strip().lower()
    if algo == "ips-grpo" and use_gpu:
        agent_filename, module_name, class_name = (
            "ips-grpo-gpu.py",
            "ips_grpo_agent_gpu",
            "IPSGRPOAgent",
        )
    elif algo in {"reinforce", "ips-reinforce"}:
        agent_filename, module_name, class_name = (
            "reinforce.py",
            "reinforce_agent",
            "ReinforceAgent",
        )
    else:
        spec = AGENT_SPECS.get(algo)
        if spec is None:
            raise ValueError(f"Unknown algo: {algo}")
        agent_filename, module_name, class_name = spec

    agent_file = Path(__file__).resolve().parent / "agent" / agent_filename
    spec = importlib.util.spec_from_file_location(module_name, agent_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {agent_file}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def resolve_runtime_device(requested_device: str, torch_module) -> str:
    requested = str(requested_device).lower()
    return requested if requested.startswith("cuda") and torch_module.cuda.is_available() else "cpu"


def common_agent_kwargs(args, env: GridEnv, runtime_device: str) -> dict[str, Any]:
    return {
        "obs_dim": env.horizon * env.num_dims,
        "action_dim": env.num_dims + 1,
        "lr": args.lr,
        "entropy_coef": args.entropy_coef,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "group_size": args.group_size,
        "num_groups": args.num_groups,
        "seed": args.seed,
        "train_epochs": args.train_epochs,
        "device": runtime_device,
    }


def buf_energy_ips_kwargs(args, env: GridEnv, runtime_device: str) -> dict[str, Any]:
    """All hyperparameters for the buf-energy-ips agent (kept in one place)."""
    policy_train_epochs = int(args.buf_train_epochs)
    density_train_epochs = int(args.buf_train_epochs)
    policy_lr = float(args.buf_lr)
    density_lr = float(args.buf_lr)
    if args.buf_policy_train_epochs is not None:
        policy_train_epochs = int(args.buf_policy_train_epochs)
    if args.buf_density_train_epochs is not None:
        density_train_epochs = int(args.buf_density_train_epochs)
    if args.buf_policy_lr is not None:
        policy_lr = float(args.buf_policy_lr)
    if args.buf_density_lr is not None:
        density_lr = float(args.buf_density_lr)
    return {
        "obs_dim": env.horizon * env.num_dims,
        "action_dim": env.num_dims + 1,
        "seed": args.seed,
        "device": runtime_device,
        "lr": args.buf_lr,
        "entropy_coef": args.buf_entropy_coef,
        "hidden_size": args.buf_hidden_size,
        "num_layers": args.buf_num_layers,
        "policy_head_num_layers": args.buf_policy_head_num_layers,
        "energy_head_num_layers": args.buf_energy_head_num_layers,
        "clip_ratio": args.buf_clip_ratio,
        "buffer_size": args.buf_buffer_size,
        "minibatch_size": args.buf_minibatch_size,
        "train_epochs": args.buf_train_epochs,
        "policy_train_epochs": policy_train_epochs,
        "density_train_epochs": density_train_epochs,
        "policy_lr": policy_lr,
        "density_lr": density_lr,
        "density_loss_coef": args.buf_density_loss_coef,
        "reward_scale_eps": args.buf_reward_scale_eps,
        "p_eps": args.buf_p_eps,
        "max_inverse_weight": args.buf_max_inverse_weight,
        "advantage_mode": args.buf_advantage_mode,
        "p_hat_mode": args.buf_p_hat_mode,
        "grad_clip_norm": args.buf_grad_clip_norm,
        "normalize_group_advantages": (not args.buf_no_normalize_advantages),
        "phat_timing": args.buf_phat_timing,
        "trunk_mode": args.buf_trunk_mode,
        "density_param_scope": args.buf_density_param_scope,
    }


def build_agent(args, env: GridEnv, AgentClass, runtime_device: str):
    algo = args.algo
    if algo == "buf-energy-ips":
        kwargs = buf_energy_ips_kwargs(args, env, runtime_device)
        return AgentClass(**kwargs)

    kwargs = common_agent_kwargs(args, env, runtime_device)
    if algo in {"ips-grpo", "energy-ips"}:
        kwargs.update(
            {
                "clip_ratio": args.clip_ratio,
                "reward_scale_eps": args.reward_scale_eps,
                "p_eps": args.p_eps,
                "max_inverse_weight": args.max_inverse_weight,
                "advantage_mode": args.advantage_mode,
            }
        )
        if algo == "ips-grpo" and not args.use_gpu:
            kwargs.update(
                {
                    "use_density_model": args.use_density_model,
                    "p_hat_mode": args.p_hat_mode,
                    "lambda_mix": args.lambda_mix,
                    "dynamic_mixing": args.dynamic_mixing,
                    "density_loss_coef": args.density_loss_coef,
                    "density_replay_steps": args.density_replay_steps,
                    "canonicalize_outcomes": args.canonicalize_outcomes,
                    "outcome_vocab_handling": args.outcome_vocab_handling,
                    "enable_debug_checks": (not args.disable_debug_checks),
                }
            )
        elif algo == "energy-ips":
            kwargs["density_loss_coef"] = args.density_loss_coef
            kwargs["p_hat_mode"] = args.energy_p_hat_mode
            kwargs["grad_clip_norm"] = args.energy_grad_clip_norm
            kwargs["phat_timing"] = args.energy_phat_timing
            kwargs["trunk_mode"] = args.energy_trunk_mode
            kwargs["density_param_scope"] = args.energy_density_param_scope
    elif algo in {"grpo", "ppo"}:
        kwargs["clip_ratio"] = args.clip_ratio
    elif algo == "mara":
        kwargs.update(
            {
                "clip_ratio": args.clip_ratio,
                "mara_beta": args.mara_beta,
                "mara_tau": args.mara_tau,
            }
        )
    elif algo in {"reinforce", "ips-reinforce"}:
        kwargs.update(
            {
                "reward_scale_eps": args.reward_scale_eps,
                "mode": ("ips" if algo == "ips-reinforce" else "vanilla"),
            }
        )
    else:
        kwargs.update(
            {
                "batch_size": (args.group_size * args.num_groups),
                "beta": args.flowrl_beta,
                "clip_epsilon": args.clip_ratio,
                "ref_policy": args.flowrl_ref_policy,
            }
        )
    return AgentClass(**kwargs)


def print_algo_config(args) -> None:
    if args.algo == "ips-grpo":
        print(
            "IPS config: "
            f"hidden_size={args.hidden_size}, "
            f"num_layers={args.num_layers}, "
            f"clip_ratio={args.clip_ratio:.4f}, "
            f"train_epochs={args.train_epochs}, "
            "use_density_model=False, "
            "p_hat_mode=group, "
            f"advantage_mode={args.advantage_mode}"
        )
        return

    if args.algo == "energy-ips":
        print(
            "Energy-IPS config: "
            f"hidden_size={args.hidden_size}, "
            f"num_layers={args.num_layers}, "
            f"clip_ratio={args.clip_ratio:.4f}, "
            f"train_epochs={args.train_epochs}, "
            f"density_loss_coef={args.density_loss_coef:.3e}, "
            f"grad_clip_norm={args.energy_grad_clip_norm}, "
            f"p_hat_mode={args.energy_p_hat_mode}, "
            f"phat_timing={args.energy_phat_timing}, "
            f"trunk_mode={args.energy_trunk_mode}, "
            f"density_param_scope={args.energy_density_param_scope}, "
            f"p_eps={args.p_eps}, "
            f"advantage_mode={args.advantage_mode}"
        )
        return

    if args.algo == "buf-energy-ips":
        has_phase_overrides = (
            args.buf_policy_train_epochs != args.buf_train_epochs
            or args.buf_density_train_epochs != args.buf_train_epochs
            or not np.isclose(args.buf_policy_lr, args.buf_lr)
            or not np.isclose(args.buf_density_lr, args.buf_lr)
        )
        if has_phase_overrides:
            train_epochs_str = (
                f"policy/density={args.buf_policy_train_epochs or args.buf_train_epochs}/"
                f"{args.buf_density_train_epochs or args.buf_train_epochs}"
            )
            lr_str = (
                f"policy/density={args.buf_policy_lr or args.buf_lr:.3e}/"
                f"{args.buf_density_lr or args.buf_lr:.3e}"
            )
        else:
            train_epochs_str = f"{args.buf_train_epochs}"
            lr_str = f"{args.buf_lr:.3e}"
        print(
            "Buf-Energy-IPS config: "
            f"buffer_size={args.buf_buffer_size}, "
            f"minibatch_size={args.buf_minibatch_size}, "
            f"train_epochs={train_epochs_str}, "
            f"lr={lr_str}, "
            f"entropy_coef={args.buf_entropy_coef:.4f}, "
            f"hidden_size={args.buf_hidden_size}, "
            f"trunk_layers={args.buf_num_layers}, "
            f"policy_head_layers={args.buf_policy_head_num_layers}, "
            f"energy_head_layers={args.buf_energy_head_num_layers}, "
            f"clip_ratio={args.buf_clip_ratio:.4f}, "
            f"density_loss_coef={args.buf_density_loss_coef:.3e}, "
            f"grad_clip_norm={args.buf_grad_clip_norm}, "
            f"p_hat_mode={args.buf_p_hat_mode}, "
            f"phat_timing={args.buf_phat_timing}, "
            f"trunk_mode={args.buf_trunk_mode}, "
            f"density_param_scope={args.buf_density_param_scope}, "
            f"p_eps={args.buf_p_eps}, "
            f"advantage_mode={args.buf_advantage_mode}"
        )
        return

    if args.algo in {"grpo", "ppo"}:
        print(
            f"{args.algo.upper()} config: "
            f"group_size={args.group_size}, "
            f"num_groups={args.num_groups}, "
            f"entropy_coef={args.entropy_coef:.4f}, "
            f"clip_ratio={args.clip_ratio:.4f}, "
            f"train_epochs={args.train_epochs}, "
            f"hidden_size={args.hidden_size}, "
            f"num_layers={args.num_layers}"
        )
        return

    if args.algo == "mara":
        print(
            "MARA config: "
            f"group_size={args.group_size}, "
            f"num_groups={args.num_groups}, "
            f"entropy_coef={args.entropy_coef:.4f}, "
            f"clip_ratio={args.clip_ratio:.4f}, "
            f"train_epochs={args.train_epochs}, "
            f"hidden_size={args.hidden_size}, "
            f"num_layers={args.num_layers}, "
            f"mara_beta={args.mara_beta:.4f}, "
            f"mara_tau={args.mara_tau:.4f}"
        )
        return

    if args.algo in {"reinforce", "ips-reinforce"}:
        print(
            "REINFORCE config: "
            f"mode={'ips' if args.algo == 'ips-reinforce' else 'vanilla'}, "
            f"group_size={args.group_size}, "
            f"num_groups={args.num_groups}, "
            f"entropy_coef={args.entropy_coef:.4f}, "
            f"train_epochs={args.train_epochs}, "
            f"hidden_size={args.hidden_size}, "
            f"num_layers={args.num_layers}, "
            f"reward_scale_eps={args.reward_scale_eps:.2e}"
        )
        return

    print(
        "FlowRL config: "
        f"batch_size={args.group_size * args.num_groups}, "
        f"train_epochs={args.train_epochs}, "
        f"hidden_size={args.hidden_size}, "
        f"num_layers={args.num_layers}, "
        f"beta={args.flowrl_beta:.4f}, "
        f"clip_epsilon={args.clip_ratio:.4f}, "
        f"entropy_coef={args.entropy_coef:.4f}, "
        f"ref_policy={args.flowrl_ref_policy}"
    )


def metrics_p_hat_mode(args) -> str:
    if args.algo == "ips-grpo":
        return "group"
    if args.algo == "energy-ips":
        return str(args.energy_p_hat_mode)
    if args.algo == "buf-energy-ips":
        return str(args.buf_p_hat_mode)
    return "n/a"


def set_seed(seed: int, torch_module) -> None:
    np.random.seed(seed)
    torch_module.manual_seed(seed)


def sample_terminal_distribution(
    agent,
    env: GridEnv,
    num_episodes: int,
    eval_batch_size: int = 4096,
) -> Dict[Tuple[int, ...], float]:
    counts: Counter = Counter()
    total_episodes = int(num_episodes)
    eval_batch_size = max(int(eval_batch_size), 1)

    # Use the agent's vectorized rollout implementation for fast batched eval.
    if hasattr(agent, "_sample_rollouts"):
        import inspect

        rollout_sig = inspect.signature(agent._sample_rollouts)
        uses_buffer_arg = "buffer_size" in rollout_sig.parameters
        remaining = total_episodes
        while remaining > 0:
            batch_n = min(eval_batch_size, remaining)
            if uses_buffer_arg:
                rollout = agent._sample_rollouts(env, buffer_size=batch_n)
            else:
                rollout = agent._sample_rollouts(env, group_size=batch_n, num_groups=1)
            terminal_states = rollout.outcomes.detach().cpu().numpy().astype(np.int32)
            for state in terminal_states:
                counts[tuple(state.tolist())] += 1
            remaining -= batch_n
    else:
        for _ in range(total_episodes):
            obs, _, _ = env.reset()
            done = False
            terminal_state = None
            while not done:
                action = agent.act(obs, deterministic=False)
                obs, _, done, state = env.step(action)
                terminal_state = state
            counts[tuple(np.asarray(terminal_state, dtype=np.int32).tolist())] += 1

    total = float(max(total_episodes, 1))
    return {k: v / total for k, v in counts.items()}


def terminal_state_vectors(
    env: GridEnv, empirical: Dict[Tuple[int, ...], float]
) -> Tuple[list[Tuple[int, ...]], np.ndarray, np.ndarray]:
    all_states = list(itertools.product(range(env.horizon), repeat=env.num_dims))

    reachable_states = []
    rewards = []
    for s in all_states:
        state_np = np.asarray(s, dtype=np.int32)
        parents, _ = env.get_parents(state_np, used_stop_action=False)
        is_start = int(np.sum(state_np)) == 0
        if len(parents) > 0 or is_start:
            reachable_states.append(tuple(state_np.tolist()))
            x = env.state_to_x(state_np)
            rewards.append(float(env.reward_func(x)))

    rewards_np = np.asarray(rewards, dtype=np.float32)
    target_probs = rewards_np / max(float(rewards_np.sum()), 1e-12)
    empirical_vec = np.asarray([empirical.get(s, 0.0) for s in reachable_states], dtype=np.float32)
    return reachable_states, target_probs, empirical_vec


def density_model_state_vector(
    states: list[Tuple[int, ...]],
    density_model_probs: Dict[Tuple[int, ...], float],
) -> np.ndarray:
    return np.asarray([density_model_probs.get(s, 0.0) for s in states], dtype=np.float32)


def l1_to_true_density(env: GridEnv, empirical: Dict[Tuple[int, ...], float]) -> float:
    _, target_probs, empirical_vec = terminal_state_vectors(env, empirical)
    return float(np.abs(empirical_vec - target_probs).mean())


def kl_to_true_density(
    env: GridEnv, empirical: Dict[Tuple[int, ...], float], eps: float = 1e-12
) -> float:
    _, target_probs, empirical_vec = terminal_state_vectors(env, empirical)
    target_safe = np.clip(target_probs.astype(np.float64), eps, None)
    empirical_safe = np.clip(empirical_vec.astype(np.float64), eps, None)
    target_safe /= max(float(target_safe.sum()), eps)
    empirical_safe /= max(float(empirical_safe.sum()), eps)
    # KL(empirical || target): how far sampled policy is from true reward-proportional density.
    return float(np.sum(empirical_safe * np.log(empirical_safe / target_safe)))


def _corners_mode_id(env: GridEnv, state: Tuple[int, ...]) -> int | None:
    state_np = np.asarray(state, dtype=np.int32)
    x = env.state_to_x(state_np)
    abs_x = np.abs(x)
    if not bool(np.all((abs_x > 0.6) & (abs_x < 0.8))):
        return None
    mode_id = 0
    for dim_idx, coord in enumerate(x):
        if coord > 0.0:
            mode_id |= (1 << dim_idx)
    return mode_id


def inference_mode_summary(
    env: GridEnv, empirical: Dict[Tuple[int, ...], float]
) -> Dict[str, object]:
    unique_states_visited = len(empirical)
    summary: Dict[str, object] = {
        "unique_states_visited": unique_states_visited,
        "modes_found": [],
        "modes_found_count": 0,
        "max_modes": None,
        "mode_coverage": None,
    }
    reward_fn_name = getattr(env.reward_func, "__name__", "")
    if reward_fn_name != "reward_corners":
        return summary

    max_modes = 2 ** env.num_dims
    mode_ids = sorted(
        {
            mode_id
            for state in empirical.keys()
            for mode_id in [_corners_mode_id(env, state)]
            if mode_id is not None
        }
    )
    summary["modes_found"] = mode_ids
    summary["modes_found_count"] = len(mode_ids)
    summary["max_modes"] = max_modes
    summary["mode_coverage"] = (len(mode_ids) / max_modes) if max_modes > 0 else 0.0
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train policy-learning agent on GridEnv.")

    # ------------------------------------------------------------------
    # Shared (all agents)
    # ------------------------------------------------------------------
    parser.add_argument(
        "--algo",
        type=lambda s: str(s).strip().lower(),
        default="buf-energy-ips",
        choices=["ips-grpo", "energy-ips", "buf-energy-ips", "grpo", "flowrl", "reinforce", "ips-reinforce", "ppo", "mara"],
        help="Learning method to train.",
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--num-dims", type=int, default=2)
    parser.add_argument("--reward-name", type=str, default="corners", choices=["corners", "cos_N"])
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--num-updates", type=int, default=1000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--train-l1-window", type=int, default=10000, help="Sliding window size for training-time L1 metric.")
    parser.add_argument("--last-k-steps",type=int,default=2000,help="If >0, collect all on-policy rollout outcomes sampled during the last K training steps and save a heatmap of the resulting empirical sampling distribution. Useful for visualising policies that oscillate around the target distribution (e.g. energy-ips).")

    # Shared optimization/model controls
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy-coef", type=float, default=1e-1)
    parser.add_argument("--clip-ratio", type=float, default=0.1, help="PPO/GRPO clipping ratio used by both GRPO and IPS-GRPO.")
    parser.add_argument("--train-epochs", type=int, default=1, help="Policy optimization epochs per rollout update for GRPO and IPS-GRPO.")
    parser.add_argument("--hidden-size", type=int, default=128, help="Hidden width for GRPO/IPS policy network.")
    parser.add_argument("--num-layers", type=int, default=2, help="Number of trunk layers for GRPO/IPS policy network.")
    parser.add_argument("--group-size", type=int, default=1024)
    parser.add_argument("--num-groups", type=int, default=1)
    parser.add_argument("--p-eps", type=float, default=1e-8, help="Optional epsilon floor for p_hat; defaults to --reward-scale-eps.")

    # ------------------------------------------------------------------
    # energy-ips
    # ------------------------------------------------------------------
    parser.add_argument("--density-loss-coef", type=float, default=10, help="Multiplier for energy-head loss in the joint update.")
    parser.add_argument("--energy-grad-clip-norm", type=float, default=1, help="Optional max gradient norm for Energy-IPS joint update (disabled when omitted).")
    parser.add_argument("--energy-p-hat-mode",type=str,default="unnormalised",choices=["unnormalised", "normalised"],help="Which energy-derived p_hat to use for reward scaling: exp(-E) or group-normalised exp(-E).")
    parser.add_argument("--energy-phat-timing",type=str,default="after_density_update",choices=["before_density_update", "after_density_update"],help="When to evaluate p_hat relative to the density-model update on the current rollout. 'before_density_update' (default): compute p_hat from the pre-update density model, then run a joint policy+density update (current behaviour). 'after_density_update': first train the density model on the rollout outcomes, then recompute p_hat with the updated density model, then run a policy-only update.")
    parser.add_argument("--energy-trunk-mode",type=str,default="shared",choices=["shared", "separate"],help="Policy/energy encoder architecture. 'separate' (default): policy and energy heads each have their own trunk, so density training does not perturb the policy's feature representation. 'shared': one trunk feeds both heads (original behaviour).")
    parser.add_argument("--energy-density-param-scope",type=str,default="energy_head_only",choices=["all", "energy_trunk_and_head", "energy_head_only"],help="Which parameters the density-phase optimizer updates. Only takes effect when --energy-phat-timing=after_density_update. 'all' (default): the main optimizer is reused for the density phase. With --energy-trunk-mode=separate this is effectively a no-op on policy parameters because the density loss has no gradient path to them, so 'all' is the simplest safe choice. 'energy_trunk_and_head': a dedicated optimizer that only updates the trunk feeding the energy head plus the energy head itself. 'energy_head_only': a dedicated optimizer that only updates the energy head, leaving the trunk(s) and policy head untouched during the density phase.")

    # ------------------------------------------------------------------
    # buf-energy-ips (all hyperparameters in one place)
    # ------------------------------------------------------------------
    parser.add_argument("--buf-buffer-size", type=int, default=1024, help="Buf-Energy-IPS: number of trajectories T collected per update (buffer size).")
    parser.add_argument("--buf-minibatch-size", type=int, default=1024, help="Buf-Energy-IPS: step-level mini-batch size used within each training epoch.")
    parser.add_argument("--buf-train-epochs", type=int, default=5, help="Buf-Energy-IPS: number of passes over the buffer per update.")
    parser.add_argument("--buf-policy-train-epochs", type=int, default=1, help="Buf-Energy-IPS: policy-update epochs per update (phase-specific override of --buf-train-epochs).")
    parser.add_argument("--buf-density-train-epochs", type=int, default=10, help="Buf-Energy-IPS: density-update epochs per update (phase-specific override of --buf-train-epochs).")
    parser.add_argument("--buf-lr", type=float, default=1e-4, help="Buf-Energy-IPS: learning rate.")
    parser.add_argument("--buf-policy-lr", type=float, default=1e-4, help="Buf-Energy-IPS: policy optimizer learning rate (phase-specific override of --buf-lr).")
    parser.add_argument("--buf-density-lr", type=float, default=1e-2, help="Buf-Energy-IPS: density optimizer learning rate (phase-specific override of --buf-lr).")
    parser.add_argument("--buf-entropy-coef", type=float, default=1e-4, help="Buf-Energy-IPS: entropy regularization coefficient.")
    parser.add_argument("--buf-hidden-size", type=int, default=128, help="Buf-Energy-IPS: hidden width of shared trunk and heads.")
    parser.add_argument("--buf-num-layers", type=int, default=2, help="Buf-Energy-IPS: number of trunk layers.")
    parser.add_argument("--buf-policy-head-num-layers", type=int, default=2, help="Buf-Energy-IPS: number of layers in the policy head MLP.")
    parser.add_argument("--buf-energy-head-num-layers", type=int, default=2, help="Buf-Energy-IPS: number of layers in the energy head MLP.")
    parser.add_argument("--buf-clip-ratio", type=float, default=0.1, help="Buf-Energy-IPS: PPO clipping ratio.")
    parser.add_argument("--buf-density-loss-coef", type=float, default=1, help="Buf-Energy-IPS: coefficient for the NCE energy-head loss in the joint update.")
    parser.add_argument("--buf-reward-scale-eps", type=float, default=1e-12, help="Buf-Energy-IPS: numerical floor used in normalization denominators.")
    parser.add_argument("--buf-p-eps", type=float, default=1e-4, help="Buf-Energy-IPS: epsilon floor for p_hat before inverse weighting.")
    parser.add_argument("--buf-max-inverse-weight", type=float, default=75, help="Buf-Energy-IPS: optional cap for inverse weight 1/p_hat (disabled when omitted).")
    parser.add_argument("--buf-advantage-mode", type=str, default="scale_reward_then_normalize", choices=["scale_reward_then_normalize", "normalize_reward_then_scale_advantage", "reward_only", "reward_over_phat"], help="Buf-Energy-IPS: how trajectory advantages are computed over the buffer.")
    parser.add_argument("--buf-p-hat-mode", type=str, default="unnormalised", choices=["unnormalised", "normalised"], help="Buf-Energy-IPS: use exp(-E) directly or normalise over the full buffer.")
    parser.add_argument("--buf-grad-clip-norm", type=float, default=None, help="Buf-Energy-IPS: optional max gradient norm (disabled when omitted).")
    parser.add_argument("--buf-phat-timing", type=str, default="after_density_update", choices=["before_density_update", "after_density_update"], help="Buf-Energy-IPS: when to evaluate p_hat relative to density-model updates. 'before_density_update': compute p_hat from the pre-update density model, keep it frozen, then run density and policy phases. 'after_density_update': train density model first, then recompute p_hat with the updated density model and run policy updates.")
    parser.add_argument("--buf-trunk-mode", type=str, default="shared", choices=["separate", "shared"], help="Buf-Energy-IPS policy/energy encoder architecture. 'shared': one trunk feeds both heads (original behaviour). 'separate': policy and energy heads use independent trunks.")
    parser.add_argument("--buf-density-param-scope", type=str, default="energy_head_only", choices=["all", "energy_trunk_and_head", "energy_head_only"], help="Buf-Energy-IPS density-phase parameter scope. 'all': update all model parameters during density phase. 'energy_trunk_and_head': update only the trunk feeding the energy head plus the energy head. 'energy_head_only': update only the energy head.")
    parser.add_argument("--buf-no-normalize-advantages", action="store_true", default=True, help="Buf-Energy-IPS: disable buffer-level normalisation of advantages in scale_reward_then_normalize mode.")

    # ------------------------------------------------------------------
    # ips-grpo
    # ------------------------------------------------------------------
    parser.add_argument("--density-replay-steps",type=int,default=0, help="Number of previous updates' terminal-outcome batches kept for density-model training (K=0 uses only current update).")
    parser.add_argument("--use-density-model", action="store_true", default=False, help="Enable auxiliary energy-based terminal-state model trained with NCE.")
    parser.add_argument("--p-hat-mode", type=str, default="group", choices=["group", "model", "mixed"], help="Denominator estimator mode for IPS scaling.")
    parser.add_argument("--advantage-mode", type=str, default="scale_reward_then_normalize", choices=["scale_reward_then_normalize", "normalize_reward_then_scale_advantage", "reward_only", "reward_over_phat"], help="How trajectory advantages are computed: scale+normalize, normalize-then-scale, reward-only, or reward/p_hat.")
    parser.add_argument("--lambda-mix", type=float, default=0, help="Mix coefficient for p_hat in mixed mode.")
    parser.add_argument("--dynamic-mixing", action="store_true", default=False, help="If set, linearly ramp lambda_mix from 0.0 to 1.0 over the first half of training, then keep it at 1.0.")
    parser.add_argument("--reward-scale-eps", type=float, default=1e-12)
    parser.add_argument("--max-inverse-weight", type=float, default=None, help="Optional cap for inverse weight 1 / max(p_hat, eps).")
    parser.add_argument("--canonicalize-outcomes", action="store_true", help="Apply outcome canonicalization before reward/density (identity for grid now).")
    parser.add_argument("--outcome-vocab-handling", type=str, default="fixed_grid", choices=["fixed_grid"], help="Outcome-ID strategy used by density estimator.")
    parser.add_argument("--disable-debug-checks", action="store_true", help="Disable IPS-GRPO-D numerical/gradient safety assertions.")
    parser.add_argument("--use-gpu",action="store_true",default=False,help="Use GPU-specific IPS-GRPO implementation (group-based learning only).")

    # ------------------------------------------------------------------
    # grpo / ppo
    # ------------------------------------------------------------------
    # (Uses shared controls above: clip-ratio, train-epochs, hidden-size,
    #  num-layers, group-size, num-groups, lr, entropy-coef.)

    # ------------------------------------------------------------------
    # reinforce / ips-reinforce
    # ------------------------------------------------------------------
    # (Uses reward-scale-eps and shared controls above.)

    # ------------------------------------------------------------------
    # flowrl
    # ------------------------------------------------------------------
    parser.add_argument(
        "--flowrl-beta", type=float, default=1.0, help="FlowRL reward-scaling coefficient."
    )
    parser.add_argument("--flowrl-clip-epsilon", type=float, default=0.2, help="FlowRL importance weight clipping.")
    parser.add_argument(
        "--flowrl-train-epochs",
        type=int,
        default=4,
        help="Number of optimization epochs over one rollout batch (enables off-policy IS updates).",
    )
    parser.add_argument(
        "--flowrl-entropy-coef",
        type=float,
        default=0.0,
        help="Optional entropy regularization for FlowRL (paper objective uses 0.0).",
    )
    parser.add_argument(
        "--flowrl-ref-policy",
        type=str,
        default="none",
        choices=["none", "random"],
        help="FlowRL reference policy: none (default) or random frozen policy.",
    )
    parser.add_argument(
        "--flowrl-use-ref-model",
        dest="flowrl_ref_policy",
        action="store_const",
        const="random",
        help="Legacy alias: use random frozen reference policy.",
    )
    parser.add_argument(
        "--no-flowrl-use-ref-model",
        dest="flowrl_ref_policy",
        action="store_const",
        const="none",
        help="Legacy alias: disable FlowRL reference-model term.",
    )

    # ------------------------------------------------------------------
    # mara
    # ------------------------------------------------------------------
    parser.add_argument(
        "--mara-beta",
        type=float,
        default=0.1,
        help="MARA reward-augmentation coefficient beta.",
    )
    parser.add_argument(
        "--mara-tau",
        type=float,
        default=1.5,
        help="MARA reward threshold tau for selecting high-quality trajectories.",
    )

    # ------------------------------------------------------------------
    # Evaluation / outputs / runtime
    # ------------------------------------------------------------------
    parser.add_argument("--eval-episodes", type=int, default=2000)
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=20000,
        help="Number of trajectories sampled per vectorized evaluation batch.",
    )
    parser.add_argument("--plot-path", type=str, default=None, help="Optional output path for diagnostic plot image.")
    parser.add_argument("--metrics-path", type=str, default=None, help="Optional output path for per-run comparative metrics (.npz).")
    parser.add_argument("--plot-top-k", type=int, default=15, help="Top-k terminal states shown in bar comparison plot.")
    parser.add_argument("--show-plot", action="store_true", help="Display plot window in addition to saving image.")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for training (e.g. cpu, cuda, cuda:0, cuda:6).",
    )
    args = parser.parse_args()



    import torch

    set_seed(args.seed, torch)

    env = GridEnv(
        horizon=args.horizon,
        num_dims=args.num_dims,
        reward_name=args.reward_name,
    )

    if args.use_gpu and args.algo != "ips-grpo":
        print("Ignoring --use-gpu because it only applies to --algo ips-grpo.")
    AgentClass = load_agent_class(algo=args.algo, use_gpu=args.use_gpu and args.algo == "ips-grpo")
    runtime_device = resolve_runtime_device(args.device, torch)
    agent = build_agent(args, env, AgentClass, runtime_device)

    if args.algo == "ips-grpo" and args.use_gpu:
        print(f"Using GPU runtime flag. Active torch device: {runtime_device}.")
        print("GPU agent runs with group-based learning only (density model disabled).")
    else:
        print(f"Using runtime device: {runtime_device}.")
    print_algo_config(args)
    print("Starting training...")
    train_kwargs: Dict[str, Any] = {
        "env": env,
        "num_updates": args.num_updates,
        "log_every": args.log_every,
        "l1_window": args.train_l1_window,
    }
    import inspect

    train_sig = inspect.signature(agent.train)
    if "last_k_steps" in train_sig.parameters and args.last_k_steps > 0:
        train_kwargs["last_k_steps"] = int(args.last_k_steps)
    elif args.last_k_steps > 0:
        print(
            f"Agent '{args.algo}' does not support --last-k-steps; "
            "skipping last-k sampling heatmap."
        )
    history = agent.train(**train_kwargs)

    print("Running evaluation...")
    empirical = sample_terminal_distribution(
        agent,
        env,
        num_episodes=args.eval_episodes,
        eval_batch_size=args.eval_batch_size,
    )
    density_model_probs = (
        agent.terminal_density_distribution(env)
        if hasattr(agent, "terminal_density_distribution")
        else {}
    )
    l1 = l1_to_true_density(env, empirical)
    kl = kl_to_true_density(env, empirical)
    states, target_vec, empirical_vec = terminal_state_vectors(env, empirical)
    mode_summary = inference_mode_summary(env, empirical)
    print(f"Final L1 to true reward-proportional density: {l1:.6f}")
    print(f"Final KL(empirical || true) to reward-proportional density: {kl:.6f}")
    print(f"Unique states visited in eval: {mode_summary['unique_states_visited']}")
    if mode_summary["max_modes"] is not None:
        print(
            "Modes found: "
            f"{mode_summary['modes_found_count']}/{mode_summary['max_modes']} "
            f"(coverage={float(mode_summary['mode_coverage']):.3f})"
        )
        print(f"Mode IDs found: {mode_summary['modes_found']}")
    else:
        print("Mode coverage is only defined for reward_name='corners'.")
    plot_dlc = args.buf_density_loss_coef if args.algo == "buf-energy-ips" else args.density_loss_coef
    plot_max_inverse_weight = (
        args.buf_max_inverse_weight if args.algo == "buf-energy-ips" else args.max_inverse_weight
    )
    plot_path = (
        Path(args.plot_path)
        if args.plot_path is not None
        else (
            Path(__file__).resolve().parent
            / f"plots_{args.algo}_groupSize_{args.group_size}_steps_{args.num_updates}.png"
        )
    )
    plot_results(
        env=env,
        history=history,
        empirical=empirical,
        density_model_probs=density_model_probs,
        save_path=plot_path,
        top_k=max(1, args.plot_top_k),
        show=args.show_plot,
        final_l1=l1,
    )

    if (
        args.last_k_steps > 0
        and hasattr(agent, "last_k_training_outcomes")
        and len(getattr(agent, "last_k_training_outcomes", [])) > 0
    ):
        last_k_outcomes = list(agent.last_k_training_outcomes)
        counts = Counter(last_k_outcomes)
        total = float(len(last_k_outcomes))
        sampling_empirical = {k: v / total for k, v in counts.items()}
        heatmap_path = plot_path.with_name(
            f"{plot_path.stem}_sampling_last_{args.last_k_steps}steps{plot_path.suffix}"
        )
        first_step = getattr(agent, "last_k_training_first_step", None)
        plot_sampling_heatmap(
            env=env,
            empirical=sampling_empirical,
            save_path=heatmap_path,
            last_k_steps=int(args.last_k_steps),
            num_samples=int(total),
            first_step=int(first_step) if first_step is not None else None,
            last_step=int(args.num_updates),
            show=args.show_plot,
        )

    if args.metrics_path is not None:
        metrics_path = Path(args.metrics_path)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        history_steps = np.asarray([int(row.get("step", 0)) for row in history], dtype=np.int32)
        history_l1 = np.asarray([float(row.get("l1_loss", np.nan)) for row in history], dtype=np.float32)
        history_grad_norm = np.asarray(
            [float(row.get("grad_norm", np.nan)) for row in history], dtype=np.float32
        )
        history_grad_max_abs = np.asarray(
            [float(row.get("grad_max_abs", np.nan)) for row in history], dtype=np.float32
        )
        density_model_vec = density_model_state_vector(states, density_model_probs)
        np.savez_compressed(
            metrics_path,
            states=np.asarray(states, dtype=np.int16),
            target=target_vec.astype(np.float32),
            empirical=empirical_vec.astype(np.float32),
            density_model=density_model_vec,
            history_steps=history_steps,
            history_l1=history_l1,
            history_grad_norm=history_grad_norm,
            history_grad_max_abs=history_grad_max_abs,
            horizon=np.int32(env.horizon),
            num_dims=np.int32(env.num_dims),
            method=np.asarray(args.algo),
            p_hat_mode=np.asarray(metrics_p_hat_mode(args)),
            group_size=np.int32(
                args.buf_buffer_size if args.algo == "buf-energy-ips" else args.group_size
            ),
            advantage_mode=np.asarray(
                args.buf_advantage_mode if args.algo == "buf-energy-ips" else args.advantage_mode
            ),
            l1_to_true_density=np.float32(l1),
            kl_empirical_to_true_density=np.float32(kl),
            unique_states_visited=np.int32(mode_summary["unique_states_visited"]),
            mode_coverage=np.float32(
                np.nan
                if mode_summary["mode_coverage"] is None
                else float(mode_summary["mode_coverage"])
            ),
            modes_found=np.asarray(mode_summary["modes_found"], dtype=np.int16),
            modes_found_count=np.int32(mode_summary["modes_found_count"]),
            max_modes=np.int32(-1 if mode_summary["max_modes"] is None else mode_summary["max_modes"]),
            run_name=np.asarray(plot_path.stem),
        )
        print(f"Saved comparative metrics to: {metrics_path}")


if __name__ == "__main__":
    main()
