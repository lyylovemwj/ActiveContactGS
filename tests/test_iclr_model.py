import pytest
import torch

from active_contact_gs.iclr_model import ICLRPhysicsPosterior
from active_contact_gs.iclr_world import ICLRContactWorld


@pytest.mark.parametrize("representation", ["flat", "independent", "autoregressive", "structured"])
def test_iclr_posterior_shapes_and_loss(representation: str) -> None:
    model = ICLRPhysicsPosterior(
        width=32, heads=4, layers=1, dropout=0.0, representation=representation
    )
    batch, probes = 3, 2
    actions = torch.randn(batch, probes, 4)
    observations = torch.randn(batch, probes, 19, 4)
    geometry = torch.tensor([[0.14, 0.06, 0.1, 0.6]]).expand(batch, -1)
    mask = torch.tensor([[True, True], [True, False], [False, False]])
    posterior = model(actions, observations, geometry, mask)
    assert posterior.joint_logits.shape == (batch, 243)
    assert posterior.factor_logits.shape == (batch, 5, 3)
    assert posterior.conditional_parameter_mean.shape == (batch, 243, 4)
    assert torch.allclose(posterior.joint_probabilities.sum(dim=-1), torch.ones(batch), atol=1e-5)
    parameters = torch.tensor([[1.0, 0.2, 0.5, 1.0]]).expand(batch, -1)
    factors = torch.tensor([[0, 1, 2, 0, 1], [1, 1, 1, 1, 1], [2, 0, 2, 0, 2]])
    loss, metrics = model.loss(posterior, parameters, factors)
    assert torch.isfinite(loss)
    assert set(metrics) >= {"exact_accuracy", "factor_accuracy", "normalized_mae"}
