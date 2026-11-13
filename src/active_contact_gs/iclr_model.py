from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from .iclr_world import (
    FACTOR_COUNT,
    FACTOR_STATES,
    WORLD_COUNT,
    all_factor_tuples,
    factors_to_index,
    normalize_parameters,
)


@dataclass
class ICLRPosteriorOutput:
    joint_logits: torch.Tensor
    factor_logits: torch.Tensor
    conditional_parameter_mean: torch.Tensor
    conditional_parameter_std: torch.Tensor

    @property
    def joint_probabilities(self) -> torch.Tensor:
        return self.joint_logits.softmax(dim=-1)

    @property
    def parameter_mean(self) -> torch.Tensor:
        return (self.joint_probabilities[..., None] * self.conditional_parameter_mean).sum(dim=-2)

    @property
    def parameter_std(self) -> torch.Tensor:
        weights = self.joint_probabilities[..., None]
        second = (weights * (self.conditional_parameter_std.square() + self.conditional_parameter_mean.square())).sum(dim=-2)
        return (second - self.parameter_mean.square()).clamp_min(1e-6).sqrt()

    def entropy(self) -> torch.Tensor:
        probability = self.joint_probabilities
        discrete = -(probability * probability.clamp_min(1e-9).log()).sum(dim=-1)
        gaussian = (0.5 * torch.log(2 * torch.pi * torch.e * self.conditional_parameter_std.square())).sum(dim=-1)
        return discrete + (probability * gaussian).sum(dim=-1)


class ICLRPhysicsPosterior(nn.Module):
    """Shared encoder with four comparable structured posterior parameterizations."""

    REPRESENTATIONS = {"flat", "independent", "autoregressive", "structured"}

    def __init__(
        self,
        *,
        observation_frames: int = 19,
        width: int = 256,
        heads: int = 8,
        layers: int = 6,
        dropout: float = 0.05,
        representation: str = "independent",
    ) -> None:
        super().__init__()
        if representation not in self.REPRESENTATIONS:
            raise ValueError(f"unknown representation: {representation}")
        self.observation_frames = observation_frames
        self.width = width
        self.representation = representation
        self.probe_encoder = nn.Sequential(
            nn.Linear(4 + observation_frames * 4, width),
            nn.GELU(),
            nn.LayerNorm(width),
            nn.Linear(width, width),
        )
        self.geometry_encoder = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
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
        self.factor_head = nn.Linear(width, FACTOR_COUNT * FACTOR_STATES)
        self.flat_head = nn.Linear(width, WORLD_COUNT)
        self.state_embedding = nn.Parameter(torch.randn(FACTOR_COUNT, FACTOR_STATES, width) * 0.02)
        self.autoregressive_heads = nn.ModuleList(nn.Linear(width, FACTOR_STATES) for _ in range(FACTOR_COUNT))
        self.pair_left = nn.Parameter(torch.randn(FACTOR_COUNT, FACTOR_STATES, 12) * 0.05)
        self.pair_right = nn.Parameter(torch.randn(FACTOR_COUNT, FACTOR_STATES, 12) * 0.05)
        self.parameter_mean_head = nn.Linear(width, 4)
        self.parameter_std_head = nn.Linear(width, 4)
        self.parameter_tuple_delta = nn.Linear(width, 8, bias=False)

    def encode(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        geometry: torch.Tensor,
        probe_mask: torch.Tensor,
    ) -> torch.Tensor:
        if observations.shape[-2:] != (self.observation_frames, 4):
            raise ValueError("unexpected observation shape")
        action_scale = torch.tensor((6.5, 6.5, 0.05, 0.05), device=actions.device)
        normalized_actions = actions / action_scale
        normalized_observations = observations.clone()
        normalized_observations[..., :2] /= geometry[:, None, None, 3:4]
        geometry_scale = torch.tensor((0.20, 0.20, 0.80, 0.75), device=geometry.device)
        probe_input = torch.cat((normalized_actions, normalized_observations.flatten(2)), dim=-1)
        probe_tokens = self.probe_encoder(probe_input)
        batch = actions.shape[0]
        tokens = torch.cat(
            (
                self.summary_token.expand(batch, -1, -1),
                self.geometry_encoder(geometry / geometry_scale).unsqueeze(1),
                probe_tokens,
            ),
            dim=1,
        )
        prefix = torch.zeros((batch, 2), device=actions.device, dtype=torch.bool)
        padding = torch.cat((prefix, ~probe_mask.bool()), dim=1)
        return self.final_norm(self.transformer(tokens, src_key_padding_mask=padding)[:, 0])

    def _joint_logits(self, summary: torch.Tensor, factor_logits: torch.Tensor) -> torch.Tensor:
        tuples = all_factor_tuples(device=summary.device)
        if self.representation == "flat":
            return self.flat_head(summary)
        unary = factor_logits.log_softmax(dim=-1)
        gathered = torch.stack(
            [unary[:, factor, tuples[:, factor]] for factor in range(FACTOR_COUNT)], dim=-1
        ).sum(dim=-1)
        if self.representation == "independent":
            return gathered
        if self.representation == "structured":
            pair_score = torch.zeros(WORLD_COUNT, device=summary.device)
            for left in range(FACTOR_COUNT):
                for right in range(left + 1, FACTOR_COUNT):
                    pair_score += (
                        self.pair_left[left, tuples[:, left]]
                        * self.pair_right[right, tuples[:, right]]
                    ).sum(dim=-1)
            return gathered + pair_score[None]
        context = torch.zeros((summary.shape[0], WORLD_COUNT, self.width), device=summary.device)
        log_probability = torch.zeros((summary.shape[0], WORLD_COUNT), device=summary.device)
        for factor, head in enumerate(self.autoregressive_heads):
            logits = head(summary[:, None, :] + context)
            state = tuples[:, factor]
            log_probability += logits.log_softmax(dim=-1)[:, torch.arange(WORLD_COUNT, device=summary.device), state]
            context = context + self.state_embedding[factor, state][None]
        return log_probability

    def forward(
        self,
        actions: torch.Tensor,
        observations: torch.Tensor,
        geometry: torch.Tensor,
        probe_mask: torch.Tensor,
    ) -> ICLRPosteriorOutput:
        summary = self.encode(actions, observations, geometry, probe_mask)
        factor_logits = self.factor_head(summary).reshape(-1, FACTOR_COUNT, FACTOR_STATES)
        joint_logits = self._joint_logits(summary, factor_logits)
        tuples = all_factor_tuples(device=summary.device)
        tuple_feature = self.state_embedding[
            torch.arange(FACTOR_COUNT, device=summary.device)[None], tuples
        ].mean(dim=1)
        delta = self.parameter_tuple_delta(tuple_feature)
        base_mean = self.parameter_mean_head(summary)[:, None]
        base_std = self.parameter_std_head(summary)[:, None]
        mean = torch.sigmoid(base_mean + delta[None, :, :4])
        std = 0.01 + 0.36 * torch.sigmoid(base_std + delta[None, :, 4:])
        return ICLRPosteriorOutput(joint_logits, factor_logits, mean, std)

    def loss(
        self,
        posterior: ICLRPosteriorOutput,
        parameters: torch.Tensor,
        factors: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        index = factors_to_index(factors)
        row = torch.arange(len(index), device=index.device)
        target = normalize_parameters(parameters)
        mean = posterior.conditional_parameter_mean[row, index]
        std = posterior.conditional_parameter_std[row, index]
        parameter_nll = (0.5 * ((target - mean) / std).square() + std.log()).mean()
        joint_loss = F.cross_entropy(posterior.joint_logits, index)
        factor_loss = F.cross_entropy(
            posterior.factor_logits.flatten(0, 1), factors.flatten()
        )
        structure_loss = factor_loss if self.representation == "independent" else joint_loss
        total = parameter_nll + structure_loss
        with torch.no_grad():
            predicted = posterior.joint_logits.argmax(dim=-1)
            predicted_factors = all_factor_tuples(device=factors.device)[predicted]
            metrics = {
                "loss": total.detach(),
                "parameter_nll": parameter_nll.detach(),
                "structure_loss": structure_loss.detach(),
                "exact_accuracy": (predicted == index).float().mean(),
                "factor_accuracy": (predicted_factors == factors).float().mean(),
                "normalized_mae": (posterior.parameter_mean - target).abs().mean(),
            }
        return total, metrics


def posterior_metrics(
    posterior: ICLRPosteriorOutput,
    parameters: torch.Tensor,
    factors: torch.Tensor,
) -> dict[str, torch.Tensor]:
    target = normalize_parameters(parameters)
    truth = factors_to_index(factors)
    predicted = posterior.joint_logits.argmax(dim=-1)
    predicted_factors = all_factor_tuples(device=factors.device)[predicted]
    return {
        "exact_accuracy": (predicted == truth).float(),
        "factor_accuracy": (predicted_factors == factors).float().mean(dim=-1),
        "normalized_parameter_mae": (posterior.parameter_mean - target).abs().mean(dim=-1),
        "posterior_entropy": posterior.entropy(),
    }
