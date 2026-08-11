"""Config for marginal (backward-corrected) exact IPS-GRPO.

Subclasses ``IPSExperimentConfig`` so every existing IPS knob still works. The
only new knob is ``backward_correction``:

    backward_correction = True   ->  exact weight uses exp(-(log P_F(tau) - log P_B(tau|x)))
    backward_correction = False  ->  falls back to the standard exact weight exp(-log P_F(tau))

The second form is exactly the existing ips_grpo behaviour and is kept as an
ablation / sanity check.

Rationale: in the phylo env many join-orderings (trajectories) build the same
tree, so P_F(tau) for one sampled ordering is NOT the marginal P(x) of the
object. Subtracting log P_B(tau|x) (uniform backward = -log N(x)) converts the
single-trajectory propensity into a marginal one whose IPS fixed point is
q(x) proportional to R(x). See README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

from grpo_experiments.ips_grpo.config import IPSExperimentConfig


@dataclass
class MarginalIPSExperimentConfig(IPSExperimentConfig):
    """IPS-GRPO with an optional backward-policy correction on the exact weight."""

    backward_correction: bool = True
    """If True, use (log P_F - log P_B) as the trajectory log-prob for exact IPS."""

    # Defaults tuned for the recommendations in README.md. These only change the
    # dataclass defaults; any CLI flag still overrides them.
    ips_propensity_mode: str = "exact"
    advantage_reward_mode: str = "exp_linear"
    ips_target_ess_fraction: float | None = 0.5

    @property
    def method(self) -> str:
        # Distinct method name so run directories never collide with ips_grpo runs.
        return "marginal_ips_grpo"

    @classmethod
    def from_ips_config(
        cls,
        base: IPSExperimentConfig,
        *,
        backward_correction: bool = True,
    ) -> "MarginalIPSExperimentConfig":
        from dataclasses import fields

        kwargs = {f.name: getattr(base, f.name) for f in fields(IPSExperimentConfig)}
        return cls(backward_correction=backward_correction, **kwargs)
