import torch

from active_contact_gs.physics import PlanarRigidBodySimulator

DEVICE = "cuda"


def test_heavier_body_moves_less_for_same_impulse() -> None:
    simulator = PlanarRigidBodySimulator(steps=10)
    action = torch.tensor([0.2, 0.0, 0.0, 0.0], device=DEVICE)
    light = simulator.rollout(torch.tensor([0.5, 0.05, 0.5, 1.0], device=DEVICE), action)
    heavy = simulator.rollout(torch.tensor([2.0, 0.05, 0.5, 1.0], device=DEVICE), action)
    assert light[-1, 0] > heavy[-1, 0]


def test_off_centre_impulse_rotates_body() -> None:
    simulator = PlanarRigidBodySimulator(steps=10)
    params = torch.tensor([1.0, 0.05, 0.5, 1.0], device=DEVICE)
    centred = simulator.rollout(params, torch.tensor([0.2, 0.0, 0.0, 0.0], device=DEVICE))
    offset = simulator.rollout(params, torch.tensor([0.2, 0.0, 0.0, 0.05], device=DEVICE))
    assert torch.abs(offset[-1, 2]) > torch.abs(centred[-1, 2])
