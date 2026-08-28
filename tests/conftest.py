"""Pytest policy for CPU CI and the full CUDA validation suite."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch


GPU_TEST_FILES = frozenset(
    {
        "test_amortized.py",
        "test_analyze.py",
        "test_belief.py",
        "test_ellipsoid.py",
        "test_evaluate_amortized.py",
        "test_experiment.py",
        "test_hypothesis.py",
        "test_identifiability.py",
        "test_object_contact.py",
        "test_particle_baseline.py",
        "test_physics.py",
        "test_prepare_ycb_gaussians.py",
        "test_video_model.py",
    }
)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Mark the formal experiment tests and skip them on CPU-only runners."""

    skip_gpu = pytest.mark.skip(reason="requires a CUDA-capable PyTorch runtime")
    for item in items:
        if Path(str(item.path)).name not in GPU_TEST_FILES:
            continue
        item.add_marker(pytest.mark.gpu)
        if not torch.cuda.is_available():
            item.add_marker(skip_gpu)
