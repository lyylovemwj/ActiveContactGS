from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .amortized_world import normalize_parameters


@dataclass
class PosteriorOutput:
    hypothesis_logits: torch.Tensor
    conditional_parameter_mean: torch.Tensor
    conditional_parameter_std: torch.Tensor

    @property
    def hypothesis_probabilities(self) -> torch.Tensor:
        return self.hypothesis_logits.softmax(dim=-1)

    @property
    def parameter_mean(self) -> torch.Tensor:
        return (
            self.hypothesis_probabilities[..., None]
            * self.conditional_parameter_mean
        ).sum(dim=-2)

    @property
    def parameter_std(self) -> torch.Tensor:
        probabilities = self.hypothesis_probabilities[..., None]
        second_moment = (
            probabilities
            * (
                self.conditional_parameter_std.square()
                + self.conditional_parameter_mean.square()
            )
        ).sum(dim=-2)
        return (second_moment - self.parameter_mean.square()).clamp_min(1e-6).sqrt()

    def parameters_for_hypotheses(
        self, hypotheses: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch = torch.arange(
            hypotheses.shape[0], device=hypotheses.device
        )[:, None].expand_as(hypotheses)
        return (
            self.conditional_parameter_mean[batch, hypotheses],
            self.conditional_parameter_std[batch, hypotheses],
        )


class AmortizedPhysicsPosterior(nn.Module):
    """Permutation-invariant posterior over contact structure and physics.

    A geometry token represents the visually reconstructed Gaussian shape. Probe
    tokens encode performed interventions and their observed pose sequences. No
    positional encoding is used, making the transformer invariant to probe order.
    """

    def __init__(
        self,
        *,
        observation_frames: int = 19,
        width: int = 192,
        heads: int = 6,
        layers: int = 4,
        dropout: float = 0.05,
        geometry_mode: str = "full",
    ) -> None:
        super().__init__()
        if geometry_mode not in {"full", "isotropic", "no_shape"}:
            raise ValueError(f"unknown geometry mode: {geometry_mode}")
        self.observation_frames = observation_frames
        self.width = width
        self.geometry_mode = geometry_mode
        probe_dimension = 4 + observation_frames * 4
        self.probe_encoder = nn.Sequential(
            nn.Linear(probe_dimension, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
        )
        self.geometry_encoder = nn.Sequential(
            nn.Linear(4, width), nn.GELU(), nn.Linear(width, width)
        )
        self.summary_token = nn.Parameter(torch.randn(1, 1, width) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.final_norm = nn.LayerNorm(width)
        self.hypothesis_head = nn.Linear(width, 2)
        # Two conditional diagonal Gaussians q(theta | contact hypothesis).
        self.parameter_head = nn.Linear(width, 16)

    def _normalize_inputs(
        self,
        actions: torch.Tensor, observations: torch.Tensor, geometry: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        action_scale = torch.tensor(
            [6.5, 6.5, 0.05, 0.05], device=actions.device, dtype=actions.dtype
        )
        actions = actions / action_scale
        position_scale = geometry[:, None, None, 3:4]
        normalized_observations = observations.clone()
        normalized_observations[..., :2] = observations[..., :2] / position_scale
        if self.geometry_mode == "isotropic":
            radius = torch.sqrt(geometry[:, 0] * geometry[:, 1])
            geometry = torch.stack(
                (radius, radius, torch.zeros_like(radius), geometry[:, 3]), dim=-1
            )
        elif self.geometry_mode == "no_shape":
            geometry = torch.stack(
                (
                    torch.zeros_like(geometry[:, 0]),
                    torch.zeros_like(geometry[:, 1]),
                    torch.zeros_like(geometry[:, 2]),
                    geometry[:, 3],
                ),
                dim=-1,
            )
        geometry_scale = torch.tensor(
            [0.20, 0.20, 0.80, 0.75],
            device=geometry.device,
            dtype=geometry.dtype,
        )
        return actions, normalized_observations, geometry / geometry_scale

    def forward(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        geometry: torch.Tensor,
        probe_mask: torch.Tensor,
    ) -> PosteriorOutput:
        if observations.shape[-2:] != (self.observation_frames, 4):
            raise ValueError(
                f"expected observations [..., {self.observation_frames}, 4], "
                f"got {tuple(observations.shape)}"
            )
        actions, observations, geometry = self._normalize_inputs(
            actions, observations, geometry
        )
        batch, probes = actions.shape[:2]
        probe_input = torch.cat((actions, observations.flatten(2)), dim=-1)
        probe_tokens = self.probe_encoder(probe_input)
        summary = self.summary_token.expand(batch, -1, -1)
        geometry_token = self.geometry_encoder(geometry).unsqueeze(1)
        tokens = torch.cat((summary, geometry_token, probe_tokens), dim=1)
        prefix_mask = torch.zeros((batch, 2), device=actions.device, dtype=torch.bool)
        padding_mask = torch.cat((prefix_mask, ~probe_mask.to(torch.bool)), dim=1)
        encoded = self.transformer(tokens, src_key_padding_mask=padding_mask)
        summary = self.final_norm(encoded[:, 0])
        hypothesis_logits = self.hypothesis_head(summary)
        parameter_raw = self.parameter_head(summary)
        parameter_mean = torch.sigmoid(parameter_raw[:, :8].reshape(-1, 2, 4))
        parameter_std = 0.010 + 0.36 * torch.sigmoid(
            parameter_raw[:, 8:].reshape(-1, 2, 4)
        )
        return PosteriorOutput(hypothesis_logits, parameter_mean, parameter_std)

    @staticmethod
    def loss(
        posterior: PosteriorOutput,
        parameters: torch.Tensor,
        hypotheses: torch.Tensor,
        *,
        structure_weight: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = normalize_parameters(parameters)
        row = torch.arange(len(hypotheses), device=hypotheses.device)
        conditional_mean = posterior.conditional_parameter_mean[row, hypotheses]
        conditional_std = posterior.conditional_parameter_std[row, hypotheses]
        residual = (target - conditional_mean) / conditional_std
        parameter_nll = (
            0.5 * residual.square() + conditional_std.log()
        ).mean()
        structure_loss = F.cross_entropy(posterior.hypothesis_logits, hypotheses)
        total = parameter_nll + structure_weight * structure_loss
        with torch.no_grad():
            normalized_mae = (target - posterior.parameter_mean).abs().mean()
            accuracy = (
                posterior.hypothesis_logits.argmax(dim=-1) == hypotheses
            ).float().mean()
        return total, {
            "loss": total.detach(),
            "parameter_nll": parameter_nll.detach(),
            "structure_loss": structure_loss.detach(),
            "normalized_mae": normalized_mae,
            "structure_accuracy": accuracy,
        }


def posterior_metrics(
    posterior: PosteriorOutput,
    parameters: torch.Tensor,
    hypotheses: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target = normalize_parameters(parameters)
    error = target - posterior.parameter_mean
    probabilities = posterior.hypothesis_logits.softmax(dim=-1)
    confidence, prediction = probabilities.max(dim=-1)
    correct = prediction == hypotheses
    bins = torch.linspace(0.0, 1.0, 11, device=parameters.device)
    ece = torch.zeros((), device=parameters.device)
    for lower, upper in zip(bins[:-1], bins[1:]):
        selected = (confidence > lower) & (confidence <= upper)
        if selected.any():
            ece = ece + selected.float().mean() * (
                correct[selected].float().mean() - confidence[selected].mean()
            ).abs()
    z90 = 1.6448536269514722
    coverage = (
        error.abs() <= z90 * posterior.parameter_std
    ).float().mean(dim=0)
    nll = (
        0.5 * (error / posterior.parameter_std).square()
        + posterior.parameter_std.log()
        + 0.5 * torch.log(torch.tensor(2.0 * torch.pi, device=parameters.device))
    ).mean(dim=0)
    return {
        "normalized_mae": error.abs().mean(dim=0),
        "normalized_rmse": error.square().mean(dim=0).sqrt(),
        "parameter_nll": nll,
        "coverage_90": coverage,
        "structure_accuracy": correct.float().mean(),
        "structure_nll": F.cross_entropy(posterior.hypothesis_logits, hypotheses),
        "structure_ece": ece,
    }
