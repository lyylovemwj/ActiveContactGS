import torch

from active_contact_gs.belief import ParticleBelief
from active_contact_gs.physics import PlanarRigidBodySimulator, default_actions

DEVICE = "cuda"


def test_observation_reduces_posterior_predictive_error() -> None:
    generator = torch.Generator(device=DEVICE).manual_seed(3)
    simulator = PlanarRigidBodySimulator()
    truth = torch.tensor([1.35, 0.27, 0.68, 1.22], device=DEVICE)
    belief = ParticleBelief.from_uniform_prior(1024, device=DEVICE, generator=generator)
    action = default_actions().to(DEVICE)[-1]
    clean = simulator.rollout(truth, action)
    predictions = simulator.rollout(belief.particles, action.expand_as(belief.particles))
    particle_error = torch.mean(torch.abs(predictions - clean.unsqueeze(0)), dim=(1, 2))
    before = torch.sum(belief.weights * particle_error)
    belief.update(simulator, action, clean, noise_std=0.004)
    after = torch.sum(belief.weights * particle_error)
    assert after < before
