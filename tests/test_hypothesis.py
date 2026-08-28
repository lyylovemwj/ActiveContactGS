import torch

from active_contact_gs.hypothesis_belief import ContactHypothesisBelief
from active_contact_gs.hypothesis_physics import ContactHypothesisSimulator, hypothesis_actions


def test_anisotropic_and_sphere_hypotheses_diverge() -> None:
    simulator = ContactHypothesisSimulator()
    params = torch.tensor([[1.0, 0.1, 0.8, 1.0], [1.0, 0.1, 0.8, 1.0]], device="cuda")
    hypotheses = torch.tensor([0, 1], device="cuda")
    action = torch.tensor([[6.2, 0.0, 0.0, 0.035], [6.2, 0.0, 0.0, 0.035]], device="cuda")
    rollout = simulator.rollout(params, hypotheses, action)
    assert (rollout[0] - rollout[1]).abs().max() > 1e-3


def test_hypothesis_prior_is_balanced_and_scores_are_finite() -> None:
    generator = torch.Generator(device="cuda").manual_seed(29)
    belief = ContactHypothesisBelief.from_prior(512, device="cuda", generator=generator)
    assert torch.allclose(belief.sphere_probability(), torch.tensor(0.5, device="cuda"))
    scores = belief.action_scores(
        ContactHypothesisSimulator(), hypothesis_actions().to("cuda"), noise_std=0.008
    )
    assert torch.isfinite(scores).all()
    assert scores.std() > 0
