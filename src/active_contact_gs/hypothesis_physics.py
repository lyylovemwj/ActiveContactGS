from __future__ import annotations

import torch


class ContactHypothesisSimulator:
    """Batched planar dynamics with ellipse-vs-sphere collision hypotheses."""

    def __init__(
        self,
        *,
        dt: float = 1.0 / 60.0,
        steps: int = 90,
        semi_axes: tuple[float, float] = (0.14, 0.045),
        arena_half_extent: float = 0.65,
        gravity: float = 9.81,
        observation_stride: int = 5,
    ) -> None:
        self.dt = dt
        self.steps = steps
        self.semi_axes = semi_axes
        self.arena_half_extent = arena_half_extent
        self.gravity = gravity
        self.observation_stride = observation_stride

    def _support(
        self, angle: torch.Tensor, sphere_hypothesis: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        a, b = self.semi_axes
        cosine, sine = torch.cos(angle), torch.sin(angle)
        radius_x = torch.sqrt((a * cosine).square() + (b * sine).square())
        radius_y = torch.sqrt((a * sine).square() + (b * cosine).square())
        # Tangential coordinates of the ellipse support points for +x and +y.
        contact_x_y = ((a * a - b * b) * sine * cosine) / radius_x.clamp_min(1e-9)
        contact_y_x = ((a * a - b * b) * sine * cosine) / radius_y.clamp_min(1e-9)
        sphere_radius = (a * b) ** 0.5
        sphere_hypothesis = sphere_hypothesis.to(torch.bool)
        radius_x = torch.where(sphere_hypothesis, sphere_radius, radius_x)
        radius_y = torch.where(sphere_hypothesis, sphere_radius, radius_y)
        contact_x_y = torch.where(sphere_hypothesis, 0.0, contact_x_y)
        contact_y_x = torch.where(sphere_hypothesis, 0.0, contact_y_x)
        return radius_x, radius_y, contact_x_y, contact_y_x

    def rollout(
        self,
        params: torch.Tensor,
        hypotheses: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        if params.shape[-1] != 4 or action.shape[-1] != 4:
            raise ValueError("params and action must end in four values")
        params, action = torch.broadcast_tensors(params, action)
        hypotheses = torch.broadcast_to(hypotheses, params.shape[:-1])
        mass, friction, restitution, inertia_scale = params.unbind(-1)
        impulse, offset = action[..., :2], action[..., 2:]
        position = torch.zeros_like(impulse)
        velocity = impulse / mass.unsqueeze(-1)
        angle = torch.full_like(mass, 0.18)
        a, b = self.semi_axes
        base_inertia = mass * (a * a + b * b) / 4
        inertia = base_inertia * inertia_scale
        torque_impulse = offset[..., 0] * impulse[..., 1] - offset[..., 1] * impulse[..., 0]
        omega = torque_impulse / inertia.clamp_min(1e-7)
        frames = []

        for step in range(self.steps):
            speed = torch.linalg.vector_norm(velocity, dim=-1, keepdim=True)
            deceleration = friction.unsqueeze(-1) * self.gravity * self.dt
            velocity = velocity * torch.clamp(
                1.0 - deceleration / speed.clamp_min(1e-6), min=0.0
            )
            omega = torch.sign(omega) * torch.clamp(
                omega.abs() - 0.28 * friction * self.gravity * self.dt / a,
                min=0.0,
            )
            position = position + velocity * self.dt
            angle = angle + omega * self.dt

            radius_x, radius_y, contact_x_y, contact_y_x = self._support(
                angle, hypotheses
            )
            limit_x = self.arena_half_extent - radius_x
            limit_y = self.arena_half_extent - radius_y
            hit_x = position[..., 0].abs() > limit_x
            hit_y = position[..., 1].abs() > limit_y
            pre_velocity = velocity
            position_x = torch.where(
                hit_x,
                position[..., 0].sign() * (2 * limit_x - position[..., 0].abs()),
                position[..., 0],
            )
            position_y = torch.where(
                hit_y,
                position[..., 1].sign() * (2 * limit_y - position[..., 1].abs()),
                position[..., 1],
            )
            reflected_x = torch.where(hit_x, -restitution * pre_velocity[..., 0], pre_velocity[..., 0])
            reflected_y = torch.where(hit_y, -restitution * pre_velocity[..., 1], pre_velocity[..., 1])
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
        params: torch.Tensor,
        hypothesis: torch.Tensor,
        action: torch.Tensor,
        *,
        noise_std: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        clean = self.rollout(params, hypothesis, action)
        noise = torch.randn(
            clean.shape,
            device=clean.device,
            dtype=clean.dtype,
            generator=generator,
        )
        return clean + noise_std * noise


def hypothesis_actions() -> torch.Tensor:
    actions = []
    for magnitude in (0.9, 2.8, 6.2):
        for direction in ((1.0, 0.0), (0.7, 0.7), (0.15, 1.0)):
            for offset_y in (-0.035, 0.0, 0.035):
                actions.append(
                    [magnitude * direction[0], magnitude * direction[1], 0.0, offset_y]
                )
    return torch.tensor(actions, dtype=torch.float32)
