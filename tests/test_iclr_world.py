import torch

from active_contact_gs.iclr_world import (
    ICLRContactWorld,
    action_direction_indices,
    all_factor_tuples,
    composition_masks,
    factors_to_index,
    iclr_actions,
    index_to_factors,
)


def test_world_index_round_trip_and_composition_split() -> None:
    tuples = all_factor_tuples()
    assert tuples.shape == (243, 5)
    assert torch.equal(index_to_factors(factors_to_index(tuples)), tuples)
    train, held_out = composition_masks()
    assert train.sum() + held_out.sum() == 243
    assert 40 <= int(held_out.sum()) <= 55
    for factor in range(5):
        assert set(tuples[train, factor].tolist()) == {0, 1, 2}


def test_action_bank_matches_paper_protocol() -> None:
    actions = iclr_actions()
    assert actions.shape == (144, 4)
    directions = action_direction_indices(actions)
    assert torch.equal(torch.bincount(directions, minlength=12), torch.full((12,), 12))


def test_contact_states_and_friction_modes_change_rollout() -> None:
    world = ICLRContactWorld(steps=15, observation_stride=5)
    parameters = torch.tensor([[1.0, 0.25, 0.7, 1.0]]).expand(4, -1)
    geometry = torch.tensor([[0.14, 0.06, 0.2, 0.55]]).expand(4, -1)
    action = torch.tensor([[6.2, 0.0, 0.0, 0.038]]).expand(4, -1)
    factors = torch.tensor(
        [[0, 0, 0, 0, 0], [1, 0, 0, 0, 0], [2, 0, 0, 0, 0], [0, 0, 0, 0, 1]]
    )
    trajectory = world.rollout(parameters, factors, action, geometry)
    assert trajectory.shape == (4, 4, 4)
    assert (trajectory[0] - trajectory[1]).abs().max() > 1e-5
    assert (trajectory[0] - trajectory[2]).abs().max() > 1e-5
    assert (trajectory[0] - trajectory[3]).abs().max() > 1e-5
