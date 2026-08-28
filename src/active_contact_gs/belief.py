from __future__ import annotations

from dataclasses import dataclass

import torch

from .physics import PlanarRigidBodySimulator

PRIOR_LOW = (0.45, 0.03, 0.15, 0.55)
PRIOR_HIGH = (2.20, 0.55, 0.92, 1.80)


@dataclass
class ParticleBelief:
    particles: torch.Tensor
    log_weights: torch.Tensor

    @classmethod
    def from_uniform_prior(
        cls,
        count: int,
        *,
        device: torch.device | str = "cpu",
        generator: torch.Generator | None = None,
    ) -> "ParticleBelief":
        lows = torch.tensor(PRIOR_LOW, device=device)
        highs = torch.tensor(PRIOR_HIGH, device=device)
        unit = torch.rand((count, 4), device=device, generator=generator)
        particles = lows + unit * (highs - lows)
        log_weights = torch.full(
            (count,),
            -torch.log(torch.tensor(float(count), device=device)),
            device=device,
        )
        return cls(particles=particles, log_weights=log_weights)

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.log_weights, dim=0)

    def mean(self) -> torch.Tensor:
        return torch.sum(self.weights[:, None] * self.particles, dim=0)

    def std(self) -> torch.Tensor:
        delta = self.particles - self.mean()
        return torch.sqrt(torch.sum(self.weights[:, None] * delta.square(), dim=0).clamp_min(0.0))

    def effective_sample_size(self) -> torch.Tensor:
        return 1.0 / self.weights.square().sum()

    def update(
        self,
        simulator: PlanarRigidBodySimulator,
        action: torch.Tensor,
        observation: torch.Tensor,
        *,
        noise_std: float,
    ) -> None:
        predictions = simulator.rollout(self.particles, action.expand_as(self.particles))
        residual = predictions - observation.unsqueeze(0)
        # Video samples are strongly correlated in time. Treating every scalar as
        # independent makes the likelihood vastly overconfident and collapses a
        # finite particle posterior onto an arbitrary near neighbour. The mean
        # squared residual is a conservative effective-sample likelihood.
        log_likelihood = -0.5 * residual.square().flatten(1).mean(dim=1) / (noise_std**2)
        self.log_weights = torch.log_softmax(self.log_weights + log_likelihood, dim=0)

    def resample_if_needed(
        self,
        *,
        threshold_fraction: float = 0.35,
        jitter_fraction: float = 0.015,
        generator: torch.Generator | None = None,
    ) -> bool:
        count = self.particles.shape[0]
        if self.effective_sample_size() >= threshold_fraction * count:
            return False
        indices = torch.multinomial(self.weights, count, replacement=True, generator=generator)
        selected = self.particles[indices]
        span = torch.tensor(PRIOR_HIGH, device=selected.device) - torch.tensor(
            PRIOR_LOW, device=selected.device
        )
        jitter = torch.randn(selected.shape, device=selected.device, generator=generator) * span * jitter_fraction
        lows = torch.tensor(PRIOR_LOW, device=selected.device)
        highs = torch.tensor(PRIOR_HIGH, device=selected.device)
        self.particles = torch.maximum(torch.minimum(selected + jitter, highs), lows)
        self.log_weights = torch.full_like(self.log_weights, -torch.log(torch.tensor(float(count), device=selected.device)))
        return True

    def action_scores(
        self,
        simulator: PlanarRigidBodySimulator,
        actions: torch.Tensor,
        *,
        noise_std: float,
        risk_penalty: float = 0.02,
        max_particles: int = 512,
    ) -> torch.Tensor:
        """Score probes by Gaussian conditional entropy reduction of parameters.

        Predictive variance alone prefers spectacular motion even when several
        physical parameters produce that motion in a confounded way.  Here the
        weighted particle ensemble estimates the joint covariance of normalized
        parameters and future observations.  Conditioning that joint Gaussian on
        a noisy observation yields the expected residual parameter covariance.
        """
        if self.particles.shape[0] > max_particles:
            idx = torch.multinomial(self.weights, max_particles, replacement=True)
            particles = self.particles[idx]
            weights = torch.full((max_particles,), 1.0 / max_particles, device=self.particles.device)
        else:
            particles = self.particles
            weights = self.weights

        parameter_span = torch.tensor(
            PRIOR_HIGH, device=particles.device, dtype=particles.dtype
        ) - torch.tensor(PRIOR_LOW, device=particles.device, dtype=particles.dtype)
        theta = particles / parameter_span
        theta_mean = torch.sum(weights[:, None] * theta, dim=0)
        theta_centered = theta - theta_mean
        theta_cov = theta_centered.T @ (weights[:, None] * theta_centered)
        eye_theta = torch.eye(theta.shape[1], device=particles.device, dtype=particles.dtype)
        _, prior_logdet = torch.linalg.slogdet(theta_cov + 1e-6 * eye_theta)

        scores = []
        for action in actions:
            pred = simulator.rollout(particles, action.expand_as(particles))
            flat = pred.flatten(1)
            pred_mean = torch.sum(weights[:, None] * flat, dim=0)
            pred_centered = flat - pred_mean
            observation_cov = pred_centered.T @ (weights[:, None] * pred_centered)
            cross_cov = theta_centered.T @ (weights[:, None] * pred_centered)
            eye_observation = torch.eye(
                flat.shape[1], device=particles.device, dtype=particles.dtype
            )
            innovation_cov = observation_cov + (noise_std**2 + 1e-6) * eye_observation
            conditional_cov = theta_cov - cross_cov @ torch.linalg.solve(
                innovation_cov, cross_cov.T
            )
            conditional_cov = 0.5 * (conditional_cov + conditional_cov.T)
            _, conditional_logdet = torch.linalg.slogdet(
                conditional_cov + 1e-5 * eye_theta
            )
            information = 0.5 * (prior_logdet - conditional_logdet)
            impulse_energy = action[:2].square().sum() / 36.0
            scores.append(information - risk_penalty * impulse_energy)
        return torch.stack(scores)
