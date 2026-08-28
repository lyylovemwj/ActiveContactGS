import torch

from active_contact_gs.analyze import paired_comparison


DEVICE = "cuda"


def test_paired_comparison_uses_ratio_of_means() -> None:
    active = torch.tensor([0.01, 0.5], device=DEVICE)
    baseline = torch.tensor([1e-9, 1.0], device=DEVICE)
    generator = torch.Generator(device=DEVICE).manual_seed(7)
    result = paired_comparison(active, baseline, bootstrap=2000, generator=generator)

    expected = ((baseline - active).mean() / baseline.mean()).item()
    assert abs(result["ratio_of_means_relative_improvement"] - expected) < 1e-6
    assert result["pairs"] == 2
    assert set(result["relative_improvement_95_ci"]) == {"lower", "median", "upper"}


def test_paired_comparison_detects_consistent_gain() -> None:
    baseline = torch.linspace(0.1, 1.0, 32, device=DEVICE)
    active = baseline * 0.6
    generator = torch.Generator(device=DEVICE).manual_seed(11)
    result = paired_comparison(active, baseline, bootstrap=4000, generator=generator)

    assert abs(result["ratio_of_means_relative_improvement"] - 0.4) < 1e-6
    assert result["relative_improvement_95_ci"]["lower"] > 0
    assert result["bootstrap_probability_active_better"] == 1.0
    assert result["active_wins"] == 32
