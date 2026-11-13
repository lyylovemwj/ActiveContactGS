from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch


PARAMETER_LOW = (0.45, 0.03, 0.15, 0.55)
PARAMETER_HIGH = (2.20, 0.55, 0.92, 1.80)
FACTOR_COUNT = 5
FACTOR_STATES = 3
WORLD_COUNT = FACTOR_STATES**FACTOR_COUNT


@dataclass(frozen=True)
class ICLRTaskBatch:
    parameters: torch.Tensor
    factors: torch.Tensor
    geometry: torch.Tensor


def all_factor_tuples(*, device: torch.device | str = "cpu") -> torch.Tensor:
    return torch.tensor(
        list(product(range(FACTOR_STATES), repeat=FACTOR_COUNT)),
        dtype=torch.long,
        device=device,
    )


def factors_to_index(factors: torch.Tensor) -> torch.Tensor:
    powers = torch.tensor((81, 27, 9, 3, 1), device=factors.device)
    return (factors.long() * powers).sum(dim=-1)


def index_to_factors(indices: torch.Tensor) -> torch.Tensor:
    values = []
    remainder = indices.long()
    for power in (81, 27, 9, 3, 1):
        values.append(torch.div(remainder, power, rounding_mode="floor"))
        remainder = remainder.remainder(power)
    return torch.stack(values, dim=-1)


def composition_masks(*, device: torch.device | str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    """Return a deterministic 80/20 tuple split with every local state in train."""
    tuples = all_factor_tuples(device=device)
    index = factors_to_index(tuples)
    # This integer hash is stable across platforms and does not leak tuple labels.
    held_out = ((index * 73 + 19) % 101) < 20
    return ~held_out, held_out


def normalize_parameters(parameters: torch.Tensor) -> torch.Tensor:
    low = torch.tensor(PARAMETER_LOW, device=parameters.device, dtype=parameters.dtype)
    high = torch.tensor(PARAMETER_HIGH, device=parameters.device, dtype=parameters.dtype)
    return (parameters - low) / (high - low)


def denormalize_parameters(parameters: torch.Tensor) -> torch.Tensor:
    low = torch.tensor(PARAMETER_LOW, device=parameters.device, dtype=parameters.dtype)
    high = torch.tensor(PARAMETER_HIGH, device=parameters.device, dtype=parameters.dtype)
    return low + parameters * (high - low)


def sample_iclr_tasks(
    count: int,
    *,
    device: torch.device | str,
    generator: torch.Generator,
    split: str,
) -> ICLRTaskBatch:
    device = torch.device(device)
    train_mask, held_out_mask = composition_masks(device=device)
    tuples = all_factor_tuples(device=device)
    if split in {"train", "id", "geometry_ood"}:
        pool = tuples[train_mask]
    elif split == "composition_ood":
        pool = tuples[held_out_mask]
    else:
        raise ValueError(f"unknown split: {split}")
    selected = torch.randint(len(pool), (count,), device=device, generator=generator)
    factors = pool[selected]
    low = torch.tensor(PARAMETER_LOW, device=device)
    high = torch.tensor(PARAMETER_HIGH, device=device)
    parameters = low + (high - low) * (
        0.03 + 0.94 * torch.rand((count, 4), device=device, generator=generator)
    )
    unit = torch.rand((count, 4), device=device, generator=generator)
    if split == "geometry_ood":
        semi_major = 0.090 + 0.100 * unit[:, 0]
        aspect = 0.20 + 0.14 * unit[:, 1]
    else:
        semi_major = 0.105 + 0.070 * unit[:, 0]
        aspect = 0.36 + 0.40 * unit[:, 1]
    geometry = torch.stack(
        (
            semi_major,
            semi_major * aspect,
            -0.75 + 1.50 * unit[:, 2],
            0.55 + 0.18 * unit[:, 3],
        ),
        dim=-1,
    )
    return ICLRTaskBatch(parameters, factors, geometry)


def iclr_actions(*, device: torch.device | str = "cpu") -> torch.Tensor:
    """The paper's 12 directions x 6 magnitudes x 2 offsets action bank."""
    actions = []
    for direction in range(12):
        angle = 2.0 * torch.pi * direction / 12
        unit = (float(torch.cos(torch.tensor(angle))), float(torch.sin(torch.tensor(angle))))
        tangent = (-unit[1], unit[0])
        for magnitude in (0.55, 0.90, 1.45, 2.30, 3.75, 6.20):
            for signed_offset in (-0.038, 0.038):
                actions.append(
                    (
                        magnitude * unit[0],
                        magnitude * unit[1],
                        signed_offset * tangent[0],
                        signed_offset * tangent[1],
                    )
                )
    return torch.tensor(actions, dtype=torch.float32, device=device)


def action_direction_indices(actions: torch.Tensor) -> torch.Tensor:
    angle = torch.atan2(actions[..., 1], actions[..., 0]).remainder(2 * torch.pi)
    return torch.round(angle * 12 / (2 * torch.pi)).long().remainder(12)


class ICLRContactWorld:
    """Vectorized planar simulator for the five-factor ICLR benchmark.

    Contact factors 0..3 correspond to +x, -x, +y, -y. State 0 uses the
    native anisotropic support; state 1 uses an area-matched rounded support;
    state 2 preserves support direction but softens impulse, tangential motion,
    and bounce. Factor 4 controls isotropic, x-grain, or y-grain friction.
    """

    def __init__(
        self,
        *,
        dt: float = 1.0 / 60.0,
        steps: int = 90,
        observation_stride: int = 5,
        gravity: float = 9.81,
        mismatch: float = 0.0,
    ) -> None:
        self.dt = dt
        self.steps = steps
        self.observation_stride = observation_stride
        self.gravity = gravity
        self.mismatch = mismatch

    @property
    def observation_frames(self) -> int:
        return (self.steps - 1) // self.observation_stride + 2

    @staticmethod
    def _direction_factor(vectors: torch.Tensor) -> torch.Tensor:
        x, y = vectors.unbind(-1)
        horizontal = x.abs() >= y.abs()
        return torch.where(
            horizontal,
            torch.where(x >= 0, 0, 1),
            torch.where(y >= 0, 2, 3),
        )

    @staticmethod
    def _native_support(
        angle: torch.Tensor, semi_major: torch.Tensor, semi_minor: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cosine, sine = torch.cos(angle), torch.sin(angle)
        major2, minor2 = semi_major.square(), semi_minor.square()
        radius_x = torch.sqrt(major2 * cosine.square() + minor2 * sine.square())
        radius_y = torch.sqrt(major2 * sine.square() + minor2 * cosine.square())
        anisotropy = major2 - minor2
        contact_x_y = anisotropy * sine * cosine / radius_x.clamp_min(1e-9)
        contact_y_x = anisotropy * sine * cosine / radius_y.clamp_min(1e-9)
        return radius_x, radius_y, contact_x_y, contact_y_x

    def rollout(
        self,
        parameters: torch.Tensor,
        factors: torch.Tensor,
        actions: torch.Tensor,
        geometry: torch.Tensor,
    ) -> torch.Tensor:
        if parameters.shape[-1] != 4 or actions.shape[-1] != 4 or geometry.shape[-1] != 4:
            raise ValueError("parameters, actions, and geometry must end in four values")
        shape = torch.broadcast_shapes(parameters.shape[:-1], actions.shape[:-1], geometry.shape[:-1], factors.shape[:-1])
        parameters = parameters.expand(*shape, 4)
        actions = actions.expand(*shape, 4)
        geometry = geometry.expand(*shape, 4)
        factors = factors.expand(*shape, FACTOR_COUNT).long()
        mass, friction, restitution, inertia_scale = parameters.unbind(-1)
        semi_major, semi_minor, initial_angle, arena = geometry.unbind(-1)
        impulse, offset = actions[..., :2], actions[..., 2:]

        launch_factor = self._direction_factor(impulse)
        launch_mode = factors.gather(-1, launch_factor[..., None]).squeeze(-1)
        compliance = (launch_mode == 2).to(parameters.dtype)
        compliance_scale = 1.0 - compliance * (0.36 + 0.20 * self.mismatch)
        effective_impulse = impulse * compliance_scale[..., None]
        position = torch.zeros_like(impulse)
        velocity = effective_impulse / mass[..., None]
        angle = initial_angle.clone()
        inertia = mass * (semi_major.square() + semi_minor.square()) / 4 * inertia_scale
        torque = offset[..., 0] * effective_impulse[..., 1] - offset[..., 1] * effective_impulse[..., 0]
        omega = torque / inertia.clamp_min(1e-7)

        friction_mode = factors[..., 4]
        grain = 0.34 + 0.25 * self.mismatch
        friction_x = friction * torch.where(friction_mode == 1, 1.0 + grain, torch.where(friction_mode == 2, 1.0 - grain, 1.0))
        friction_y = friction * torch.where(friction_mode == 2, 1.0 + grain, torch.where(friction_mode == 1, 1.0 - grain, 1.0))
        frames = []
        for step in range(self.steps):
            decrement = self.gravity * self.dt * torch.stack((friction_x, friction_y), dim=-1)
            velocity = torch.sign(velocity) * (velocity.abs() - decrement).clamp_min(0.0)
            angular_decrement = 0.28 * friction * self.gravity * self.dt / semi_major.clamp_min(1e-5)
            omega = torch.sign(omega) * (omega.abs() - angular_decrement).clamp_min(0.0)
            position = position + velocity * self.dt
            angle = angle + omega * self.dt

            radius_x, radius_y, contact_x_y, contact_y_x = self._native_support(angle, semi_major, semi_minor)
            rounded_radius = torch.sqrt(semi_major * semi_minor)
            side_x = torch.where(position[..., 0] >= 0, 0, 1)
            side_y = torch.where(position[..., 1] >= 0, 2, 3)
            mode_x = factors.gather(-1, side_x[..., None]).squeeze(-1)
            mode_y = factors.gather(-1, side_y[..., None]).squeeze(-1)
            radius_x = torch.where(mode_x == 1, rounded_radius, radius_x)
            radius_y = torch.where(mode_y == 1, rounded_radius, radius_y)
            contact_x_y = torch.where(mode_x == 1, 0.0, contact_x_y)
            contact_y_x = torch.where(mode_y == 1, 0.0, contact_y_x)
            limit_x, limit_y = arena - radius_x, arena - radius_y
            hit_x = position[..., 0].abs() > limit_x
            hit_y = position[..., 1].abs() > limit_y
            pre_velocity = velocity
            bounce_x = restitution * torch.where(mode_x == 2, 0.42 - 0.18 * self.mismatch, 1.0)
            bounce_y = restitution * torch.where(mode_y == 2, 0.42 - 0.18 * self.mismatch, 1.0)
            reflected_x = torch.where(hit_x, -bounce_x * pre_velocity[..., 0], pre_velocity[..., 0])
            reflected_y = torch.where(hit_y, -bounce_y * pre_velocity[..., 1], pre_velocity[..., 1])
            position_x = torch.where(hit_x, position[..., 0].sign() * (2 * limit_x - position[..., 0].abs()), position[..., 0])
            position_y = torch.where(hit_y, position[..., 1].sign() * (2 * limit_y - position[..., 1].abs()), position[..., 1])
            impulse_x = mass * (reflected_x - pre_velocity[..., 0])
            impulse_y = mass * (reflected_y - pre_velocity[..., 1])
            tangential_x = torch.where(mode_x == 2, 0.48 - 0.18 * self.mismatch, 1.0)
            tangential_y = torch.where(mode_y == 2, 0.48 - 0.18 * self.mismatch, 1.0)
            omega = omega + (-contact_x_y * impulse_x * tangential_x + contact_y_x * impulse_y * tangential_y) / inertia.clamp_min(1e-7)
            position = torch.stack((position_x, position_y), dim=-1)
            velocity = torch.stack((reflected_x, reflected_y), dim=-1)
            if step % self.observation_stride == 0 or step == self.steps - 1:
                frames.append(torch.stack((position[..., 0], position[..., 1], torch.sin(angle), torch.cos(angle)), dim=-1))
        return torch.stack(frames, dim=-2)

    def observe(
        self,
        tasks: ICLRTaskBatch,
        actions: torch.Tensor,
        *,
        noise_std: float,
        generator: torch.Generator,
    ) -> torch.Tensor:
        clean = self.rollout(tasks.parameters, tasks.factors, actions, tasks.geometry)
        return clean + noise_std * torch.randn(clean.shape, device=clean.device, generator=generator)
