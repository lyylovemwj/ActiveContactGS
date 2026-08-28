from __future__ import annotations

from dataclasses import dataclass

import torch

from .belief import PRIOR_HIGH, PRIOR_LOW
from .hypothesis_physics import ContactHypothesisSimulator


@dataclass
class ContactHypothesisBelief:
    particles: torch.Tensor
    hypotheses: torch.Tensor
    log_weights: torch.Tensor

    @classmethod
    def from_prior(
        cls,
        count: int,
        *,
        device: str,
        generator: torch.Generator,
    ) -> "ContactHypothesisBelief":
        low = torch.tensor(PRIOR_LOW, device=device)
        high = torch.tensor(PRIOR_HIGH, device=device)
        particles = low + torch.rand((count, 4), device=device, generator=generator) * (high - low)
        hypotheses = torch.arange(count, device=device).remainder(2)
        log_weights = torch.full((count,), -torch.log(torch.tensor(float(count), device=device)), device=device)
        return cls(particles, hypotheses, log_weights)

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.log_weights, dim=0)

    def parameter_mean(self) -> torch.Tensor:
        return (self.weights[:, None] * self.particles).sum(dim=0)

    def sphere_probability(self) -> torch.Tensor:
        return self.weights[self.hypotheses == 1].sum()

    def effective_sample_size(self) -> torch.Tensor:
        return 1 / self.weights.square().sum()

    def update(
        self,
        simulator: ContactHypothesisSimulator,
        action: torch.Tensor,
        observation: torch.Tensor,
        *,
        noise_std: float,
    ) -> None:
        prediction = simulator.rollout(
            self.particles, self.hypotheses, action.expand_as(self.particles)
        )
        residual = prediction - observation.unsqueeze(0)
        likelihood = -0.5 * residual.square().flatten(1).mean(dim=-1) / noise_std**2
        self.log_weights = torch.log_softmax(self.log_weights + likelihood, dim=0)

    def resample_if_needed(
        self,
        *,
        generator: torch.Generator,
        threshold_fraction: float = 0.35,
        jitter_fraction: float = 0.012,
    ) -> bool:
        count = self.particles.shape[0]
        if self.effective_sample_size() >= count * threshold_fraction:
            return False
        indices = torch.multinomial(
            self.weights, count, replacement=True, generator=generator
        )
        low = torch.tensor(PRIOR_LOW, device=self.particles.device)
        high = torch.tensor(PRIOR_HIGH, device=self.particles.device)
        selected = self.particles[indices]
        jitter = torch.randn(
            selected.shape, device=selected.device, generator=generator
        ) * (high - low) * jitter_fraction
        self.particles = (selected + jitter).clamp(min=low, max=high)
        self.hypotheses = self.hypotheses[indices]
        self.log_weights = torch.full_like(
            self.log_weights, -torch.log(torch.tensor(float(count), device=selected.device))
        )
        return True

    def action_scores(
        self,
        simulator: ContactHypothesisSimulator,
        actions: torch.Tensor,
        *,
        noise_std: float,
        max_particles: int = 512,
        risk_penalty: float = 0.02,
    ) -> torch.Tensor:
        if self.particles.shape[0] > max_particles:
            indices = torch.multinomial(self.weights, max_particles, replacement=True)
            particles = self.particles[indices]
            hypotheses = self.hypotheses[indices]
            weights = torch.full((max_particles,), 1 / max_particles, device=particles.device)
        else:
            particles, hypotheses, weights = self.particles, self.hypotheses, self.weights
        span = torch.tensor(PRIOR_HIGH, device=particles.device) - torch.tensor(
            PRIOR_LOW, device=particles.device
        )
        latent = torch.cat((particles / span, hypotheses[:, None].to(particles.dtype)), dim=-1)
        latent_centered = latent - (weights[:, None] * latent).sum(dim=0)
        latent_covariance = latent_centered.T @ (weights[:, None] * latent_centered)
        latent_eye = torch.eye(latent.shape[-1], device=particles.device)
        _, prior_logdet = torch.linalg.slogdet(latent_covariance + 1e-5 * latent_eye)
        scores = []
        for action in actions:
            prediction = simulator.rollout(
                particles, hypotheses, action.expand_as(particles)
            ).flatten(1)
            prediction_centered = prediction - (weights[:, None] * prediction).sum(dim=0)
            observation_covariance = prediction_centered.T @ (
                weights[:, None] * prediction_centered
            )
            cross_covariance = latent_centered.T @ (
                weights[:, None] * prediction_centered
            )
            observation_eye = torch.eye(prediction.shape[-1], device=particles.device)
            conditional = latent_covariance - cross_covariance @ torch.linalg.solve(
                observation_covariance + (noise_std**2 + 1e-6) * observation_eye,
                cross_covariance.T,
            )
            conditional = 0.5 * (conditional + conditional.T)
            _, conditional_logdet = torch.linalg.slogdet(conditional + 1e-5 * latent_eye)
            information = 0.5 * (prior_logdet - conditional_logdet)
            scores.append(information - risk_penalty * action[:2].square().sum() / 40)
        return torch.stack(scores)
