from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ProbeAction:
    """Known planar impulse applied at an offset in the object's local frame."""

    impulse_x: float
    impulse_y: float
    offset_x: float
    offset_y: float

    def tensor(self, *, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.tensor(
            [self.impulse_x, self.impulse_y, self.offset_x, self.offset_y],
            dtype=torch.float32,
            device=device,
        )


class PlanarRigidBodySimulator:
    """Small batched rigid-body simulator used to validate active identification.

    Parameter order is ``mass, friction, restitution, inertia_scale``.  The body
    is reset before every probe. Observations contain x/y position and a continuous
    angle representation (sin(theta), cos(theta)).
    """

    def __init__(
        self,
        *,
        dt: float = 1.0 / 60.0,
        steps: int = 90,
        radius: float = 0.08,
        arena_half_extent: float = 0.65,
        gravity: float = 9.81,
        observation_stride: int = 5,
    ) -> None:
        self.dt = dt
        self.steps = steps
        self.radius = radius
        self.arena_half_extent = arena_half_extent
        self.gravity = gravity
        self.observation_stride = observation_stride

    def rollout(self, params: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Return batched observations with shape ``[..., T, 4]``."""
        if params.shape[-1] != 4 or action.shape[-1] != 4:
            raise ValueError("params and action must end in four values")

        params, action = torch.broadcast_tensors(params, action)
        mass, friction, restitution, inertia_scale = params.unbind(-1)
        impulse = action[..., :2]
        offset = action[..., 2:]

        pos = torch.zeros_like(impulse)
        vel = impulse / mass.unsqueeze(-1)
        angle = torch.zeros_like(mass)

        base_inertia = 0.5 * mass * (self.radius**2)
        inertia = base_inertia * inertia_scale
        torque_impulse = offset[..., 0] * impulse[..., 1] - offset[..., 1] * impulse[..., 0]
        omega = torque_impulse / inertia.clamp_min(1e-6)

        frames: list[torch.Tensor] = []
        limit = self.arena_half_extent - self.radius
        for step in range(self.steps):
            speed = torch.linalg.vector_norm(vel, dim=-1, keepdim=True)
            decel = friction.unsqueeze(-1) * self.gravity * self.dt
            vel = vel * torch.clamp(1.0 - decel / speed.clamp_min(1e-6), min=0.0)

            angular_decel = 0.35 * friction * self.gravity * self.dt / self.radius
            omega = torch.sign(omega) * torch.clamp(torch.abs(omega) - angular_decel, min=0.0)

            pos = pos + vel * self.dt
            angle = angle + omega * self.dt

            hit = torch.abs(pos) > limit
            pos = torch.where(hit, torch.sign(pos) * (2.0 * limit - torch.abs(pos)), pos)
            vel = torch.where(hit, -restitution.unsqueeze(-1) * vel, vel)
            # Off-axis impacts exchange a small amount of translation and rotation.
            wall_torque = (hit[..., 0].to(pos.dtype) * vel[..., 1] - hit[..., 1].to(pos.dtype) * vel[..., 0])
            omega = omega + 0.12 * wall_torque / self.radius

            if step % self.observation_stride == 0 or step == self.steps - 1:
                frames.append(
                    torch.stack((pos[..., 0], pos[..., 1], torch.sin(angle), torch.cos(angle)), dim=-1)
                )

        return torch.stack(frames, dim=-2)

    def observe(
        self,
        params: torch.Tensor,
        action: torch.Tensor,
        *,
        noise_std: float,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        clean = self.rollout(params, action)
        noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
        return clean + noise_std * noise


def default_actions() -> torch.Tensor:
    """A compact action bank spanning energy, direction, and contact eccentricity."""
    actions = []
    # The strongest probes deliberately reach an arena wall across the full prior,
    # making restitution observable instead of only exciting free sliding.
    for magnitude in (0.8, 2.5, 6.0):
        for direction in ((1.0, 0.0), (0.7, 0.7), (0.2, 1.0)):
            for offset_y in (-0.02, 0.0, 0.02):
                actions.append(
                    [magnitude * direction[0], magnitude * direction[1], 0.0, offset_y]
                )
    return torch.tensor(actions, dtype=torch.float32)
