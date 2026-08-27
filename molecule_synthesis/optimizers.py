"""Optimizers used by the molecule-synthesis objective adapters."""

from __future__ import annotations

import gin
import torch
from torch import nn

from rgfn.trainer.optimizers.optimizer_base import OptimizerBase


@gin.configurable()
class MIPSOptimizer(OptimizerBase):
    """Adam with separate forward and reverse rates plus repeated reverse MLE.

    The upstream trainer performs one backward pass on the returned forward
    objective. ``step`` applies that forward update first, then refits the
    learned reverse policy on the same on-policy trajectories, matching the
    update ordering used by the phylogenetics MIPS implementation.
    """

    def __init__(
        self,
        forward_lr: float = 1e-4,
        reverse_lr: float = 1e-3,
        reverse_train_epochs: int = 4,
        reverse_grad_clip_norm: float = 1.0,
        **adam_kwargs,
    ):
        super().__init__("Adam")
        if forward_lr <= 0.0 or reverse_lr <= 0.0:
            raise ValueError("forward_lr and reverse_lr must be positive")
        if reverse_train_epochs < 1:
            raise ValueError("reverse_train_epochs must be at least one")
        if reverse_grad_clip_norm <= 0.0:
            raise ValueError("reverse_grad_clip_norm must be positive")
        self.forward_lr = float(forward_lr)
        self.reverse_lr = float(reverse_lr)
        self.reverse_train_epochs = int(reverse_train_epochs)
        self.reverse_grad_clip_norm = float(reverse_grad_clip_norm)
        self.adam_kwargs = adam_kwargs
        self.model: nn.Module | None = None
        self.reverse_parameters: list[nn.Parameter] = []

    def initialize(self, model: nn.Module):
        if not hasattr(model, "forward_policy") or not hasattr(model, "backward_policy"):
            raise TypeError("MIPSOptimizer requires forward_policy and backward_policy")
        self.model = model
        forward_parameters = list(model.forward_policy.parameters())
        forward_ids = {id(parameter) for parameter in forward_parameters}
        self.reverse_parameters = [
            parameter
            for parameter in model.backward_policy.parameters()
            if id(parameter) not in forward_ids
        ]
        if not forward_parameters or not self.reverse_parameters:
            raise ValueError("MIPSOptimizer requires distinct trainable forward and reverse parameters")
        self.optimizer = torch.optim.Adam(
            [
                {"params": forward_parameters, "lr": self.forward_lr},
                {"params": self.reverse_parameters, "lr": self.reverse_lr},
            ],
            **self.adam_kwargs,
        )

    def step(self):
        if self.model is None or not hasattr(self.model, "compute_reverse_loss"):
            raise RuntimeError("MIPSOptimizer has not been initialized with a MIPS objective")
        # Only forward parameters have gradients from the main objective.
        self.optimizer.step()
        latest_reverse_loss = None
        for _ in range(self.reverse_train_epochs):
            self.optimizer.zero_grad()
            reverse_loss = self.model.compute_reverse_loss()
            reverse_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.reverse_parameters, self.reverse_grad_clip_norm
            )
            self.optimizer.step()
            latest_reverse_loss = reverse_loss.detach()
        self.model.latest_reverse_update_loss = latest_reverse_loss
