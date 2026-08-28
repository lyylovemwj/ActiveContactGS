from __future__ import annotations

from dataclasses import dataclass

import torch

from .belief import PRIOR_HIGH, PRIOR_LOW


@dataclass(frozen=True)
class TaskBatch:
    """A batch of visually observed shapes and hidden physical worlds.

    ``geometry`` stores the rendered semi-major axis, semi-minor axis, known
    initial orientation, and arena half extent.  The binary hypothesis says
    whether contact follows the native anisotropic shape (0) or an area-matched
    isotropic proxy (1).  Thus the visual geometry is identical under both
    hypotheses while the physical contact model is not.
    """

    parameters: torch.Tensor
    hypotheses: torch.Tensor
    geometry: torch.Tensor


class AmortizedContactWorld:
    """GPU-vectorized contact-rich worlds used for cross-object learning."""

    def __init__(
        self,
        *,
        dt: float = 1.0 / 60.0,
        steps: int = 90,
        gravity: float = 9.81,
        observation_stride: int = 5,
    ) -> None:
        self.dt = dt
        self.steps = steps
        self.gravity = gravity
        self.observation_stride = observation_stride

    @property
    def observation_frames(self) -> int:
        return (self.steps - 1) // self.observation_stride + 2

    @staticmethod
    def _support(
        angle: torch.Tensor,
        semi_major: torch.Tensor,
        semi_minor: torch.Tensor,
        sphere_hypothesis: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cosine, sine = torch.cos(angle), torch.sin(angle)
        major2, minor2 = semi_major.square(), semi_minor.square()
        radius_x = torch.sqrt(major2 * cosine.square() + minor2 * sine.square())
        radius_y = torch.sqrt(major2 * sine.square() + minor2 * cosine.square())
        anisotropy = major2 - minor2
        contact_x_y = anisotropy * sine * cosine / radius_x.clamp_min(1e-9)
        contact_y_x = anisotropy * sine * cosine / radius_y.clamp_min(1e-9)
        sphere_radius = torch.sqrt(semi_major * semi_minor)
        if sphere_hypothesis.is_floating_point():
            # A continuous relaxation is used only for local identifiability and
            # Fisher-information diagnostics. End-to-end experiments use binary h.
            alpha = sphere_hypothesis.clamp(0.0, 1.0)
            radius_x = torch.lerp(radius_x, sphere_radius, alpha)
            radius_y = torch.lerp(radius_y, sphere_radius, alpha)
            contact_x_y = contact_x_y * (1.0 - alpha)
            contact_y_x = contact_y_x * (1.0 - alpha)
        else:
            sphere_hypothesis = sphere_hypothesis.to(torch.bool)
            radius_x = torch.where(sphere_hypothesis, sphere_radius, radius_x)
            radius_y = torch.where(sphere_hypothesis, sphere_radius, radius_y)
            contact_x_y = torch.where(sphere_hypothesis, 0.0, contact_x_y)
            contact_y_x = torch.where(sphere_hypothesis, 0.0, contact_y_x)
        return radius_x, radius_y, contact_x_y, contact_y_x

    def rollout(
        self,
        parameters: torch.Tensor,
        hypotheses: torch.Tensor,
        actions: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        """Simulate pose observations with arbitrary leading batch dimensions."""
        if parameters.shape[-1] != 4 or actions.shape[-1] != 4 or geometry.shape[-1] != 4:
            raise ValueError("parameters, actions, and geometry must end in four values")
        parameters, actions, geometry = torch.broadcast_tensors(
            parameters, actions, geometry
        )
        hypotheses = torch.broadcast_to(hypotheses, parameters.shape[:-1])
        mass, friction, restitution, inertia_scale = parameters.unbind(-1)
        semi_major, semi_minor, initial_angle, arena_half_extent = geometry.unbind(-1)
        impulse, offset = actions[..., :2], actions[..., 2:]
        position = torch.zeros_like(impulse)
        velocity = impulse / mass.unsqueeze(-1)
        angle = initial_angle.clone()
        base_inertia = mass * (semi_major.square() + semi_minor.square()) / 4.0
        inertia = base_inertia * inertia_scale
        torque_impulse = offset[..., 0] * impulse[..., 1] - offset[..., 1] * impulse[..., 0]
        omega = torque_impulse / inertia.clamp_min(1e-7)
        frames: list[torch.Tensor] = []

        for step in range(self.steps):
            speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
            deceleration = friction.unsqueeze(-1) * self.gravity * self.dt
            velocity = velocity * torch.clamp(
                1.0 - deceleration / speed.clamp_min(1e-6), min=0.0
            )
            omega = torch.sign(omega) * torch.clamp(
                omega.abs()
                - 0.28
                * friction
                * self.gravity
                * self.dt
                / semi_major.clamp_min(1e-5),
                min=0.0,
            )
            position = position + velocity * self.dt
            angle = angle + omega * self.dt

            radius_x, radius_y, contact_x_y, contact_y_x = self._support(
                angle, semi_major, semi_minor, hypotheses
            )
            limit_x = arena_half_extent - radius_x
            limit_y = arena_half_extent - radius_y
            hit_x = position[..., 0].abs() > limit_x
            hit_y = position[..., 1].abs() > limit_y
            pre_velocity = velocity
            position_x = torch.where(
                hit_x,
                position[..., 0].sign() * (2.0 * limit_x - position[..., 0].abs()),
                position[..., 0],
            )
            position_y = torch.where(
                hit_y,
                position[..., 1].sign() * (2.0 * limit_y - position[..., 1].abs()),
                position[..., 1],
            )
            reflected_x = torch.where(
                hit_x, -restitution * pre_velocity[..., 0], pre_velocity[..., 0]
            )
            reflected_y = torch.where(
                hit_y, -restitution * pre_velocity[..., 1], pre_velocity[..., 1]
            )
            impulse_x = mass * (reflected_x - pre_velocity[..., 0])
            impulse_y = mass * (reflected_y - pre_velocity[..., 1])
            wall_torque = -contact_x_y * impulse_x + contact_y_x * impulse_y
            omega = omega + wall_torque / inertia.clamp_min(1e-7)
            position = torch.stack((position_x, position_y), dim=-1)
            velocity = torch.stack((reflected_x, reflected_y), dim=-1)

            if step % self.observation_stride == 0 or step == self.steps - 1:
                frames.append(
                    torch.stack(
                        (
                            position[..., 0],
                            position[..., 1],
                            torch.sin(angle),
                            torch.cos(angle),
                        ),
                        dim=-1,
                    )
                )
        return torch.stack(frames, dim=-2)

    def observe(
        self,
        tasks: TaskBatch,
        actions: torch.Tensor,
        *,
        noise_std: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        clean = self.rollout(
            tasks.parameters, tasks.hypotheses, actions, tasks.geometry
        )
        return clean + noise_std * torch.randn(
            clean.shape, device=clean.device, dtype=clean.dtype, generator=generator
        )


def sample_tasks(
    count: int,
    *,
    device: torch.device | str,
    generator: torch.Generator,
    split: str = "train",
) -> TaskBatch:
    """Sample disjoint geometry distributions for ID and OOD evaluation."""
    device = torch.device(device)
    low = torch.tensor(PRIOR_LOW, device=device)
    high = torch.tensor(PRIOR_HIGH, device=device)
    parameters = low + (high - low) * (
        0.04 + 0.92 * torch.rand((count, 4), device=device, generator=generator)
    )
    hypotheses = torch.randint(2, (count,), device=device, generator=generator)
    unit = torch.rand((count, 4), device=device, generator=generator)
    if split in {"train", "id"}:
        semi_major = 0.105 + 0.070 * unit[:, 0]
        aspect = 0.36 + 0.40 * unit[:, 1]
    elif split == "ood_thin":
        semi_major = 0.090 + 0.100 * unit[:, 0]
        aspect = 0.20 + 0.14 * unit[:, 1]
    elif split == "ood_round":
        semi_major = 0.090 + 0.100 * unit[:, 0]
        aspect = 0.80 + 0.16 * unit[:, 1]
    else:
        raise ValueError(f"unknown split: {split}")
    semi_minor = semi_major * aspect
    initial_angle = -0.75 + 1.50 * unit[:, 2]
    arena = 0.55 + 0.18 * unit[:, 3]
    geometry = torch.stack((semi_major, semi_minor, initial_angle, arena), dim=-1)
    return TaskBatch(parameters, hypotheses, geometry)


def intervention_actions(*, device: torch.device | str = "cpu") -> torch.Tensor:
    """Broad, safety-bounded intervention bank with weak and contact-rich probes."""
    actions: list[list[float]] = []
    directions = (
        (1.0, 0.0),
        (0.9239, 0.3827),
        (0.7071, 0.7071),
        (0.3827, 0.9239),
        (0.0, 1.0),
        (-0.3827, 0.9239),
        (-0.7071, 0.7071),
        (-0.9239, 0.3827),
    )
    for magnitude in (0.45, 0.75, 1.15, 1.75, 2.6, 3.8, 5.2, 6.5):
        for direction_x, direction_y in directions:
            for offset_y in (-0.045, 0.0, 0.045):
                actions.append(
                    [magnitude * direction_x, magnitude * direction_y, 0.0, offset_y]
                )
    return torch.tensor(actions, dtype=torch.float32, device=device)


def normalize_parameters(parameters: torch.Tensor) -> torch.Tensor:
    low = torch.tensor(PRIOR_LOW, device=parameters.device, dtype=parameters.dtype)
    high = torch.tensor(PRIOR_HIGH, device=parameters.device, dtype=parameters.dtype)
    return (parameters - low) / (high - low)


def denormalize_parameters(normalized: torch.Tensor) -> torch.Tensor:
    low = torch.tensor(PRIOR_LOW, device=normalized.device, dtype=normalized.dtype)
    high = torch.tensor(PRIOR_HIGH, device=normalized.device, dtype=normalized.dtype)
    return low + normalized * (high - low)
