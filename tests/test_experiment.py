from active_contact_gs.experiment import run_trial


def test_random_baseline_samples_without_replacement() -> None:
    result = run_trial(
        strategy="random",
        seed=23,
        probes=8,
        particle_count=256,
        noise_std=0.008,
        device="cuda",
    )
    indices = [row["action_index"] for row in result["history"]]
    assert len(indices) == len(set(indices))
