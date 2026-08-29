# active_contact_gs — Package Overview

This directory contains the **core method, training, evaluation, and analysis code** of **ActiveContactGS**, the research project behind *Contact Geometry Is a Latent Variable: Active Bayesian Identification of Gaussian Physical Worlds*.

## What this package does

ActiveContactGS studies **active rigid-body system identification** from object-centric anisotropic Gaussian geometry. In plain terms, the system figures out the physics of an unknown object by deciding which small contact actions to take next, and then uses the resulting interaction history to update a belief about the object's latent parameters.

The package is organized around three tightly coupled ideas:
  
1. **Native Gaussian geometry.** Objects are modelled with anisotropic Gaussian ellipsoids, and contact is computed directly on that geometry instead of being approximated by spheres. This preserves contact normals far more accurately.

2. **A hybrid Bayesian posterior.** The posterior is over both the latent contact structure and the continuous physical parameters (friction, restitution, mass properties, etc.), combining pose observations and Gaussian-video models.

3. **Amortized active probing.** An amortized model selects the next action to maximise expected information gain among safe interaction candidates, instead of hand-crafting a planner.

## Main modules

| Module | Purpose |
| --- | --- |
| `physics.py` | Differentiable planar rigid-body simulation with friction, restitution, off-centre impulses, and wall contact. |
| `ellipsoid.py` | Contact computation and time-of-impact on native anisotropic Gaussian ellipsoids. |
| `object_contact.py` | Gaussian-object contact routines for anisotropic ellipsoid objects. |
| `belief.py` | Particle-based belief state over latent parameters. |
| `amortized_model.py` / `amortized_world.py` | Amortized inference/action-selection model and its training world. |
| `video_model.py` | Gaussian-video posterior model. |
| `train_amortized.py` / `generate_amortized_dataset.py` | Training the amortized model and generating the procedural dataset. |
| `evaluate_amortized.py` / `experiment.py` | Evaluation of Active/Random/Fixed policies and the paper-scale experiments. |
| `analyze.py`, `identifiability_analysis.py`, `hypothesis_*.py` | Analysis, identifiability, and hypothesis-testing utilities. |
| `benchmark_ellipsoid.py` / `benchmark_object_contact.py` | Benchmarks for contact geometry. |
| `prepare_ycb_gaussians.py` | Prepares YCB object files as Gaussian ellipsoids. |
| `plot_*.py` | Figure generation for the paper. |

## How it fits together

The data flow mirrors the paper's pipeline: a procedural dataset of contact tasks is generated, the amortized model is trained on it, and then the trained model is evaluated under **Active**, **Random**, and **Fixed** probing strategies. Analysis and plotting utilities turn the results into the figures used in the manuscript.

## Public API

`active_contact_gs.__init__` exposes the core building blocks:

- `EllipsoidContact`, `ContinuousEllipsoidContact`, `ellipsoid_contact`, `compiled_ellipsoid_contact`, `ellipsoid_time_of_impact` — contact primitives;
- `PlanarRigidBodySimulator`, `ProbeAction` — the differentiable simulator and its actions;
- `ParticleBelief` — the particle-based posterior;
- `GaussianObjectContact`, `gaussian_object_contact` — Gaussian-object contact.

For end-to-end usage and reproduction instructions, see the repository root `README.md` and `docs/`.

## Note on scope

This package contains code and figure-generation scripts only. Checkpoints, generated datasets, raw experimental results, and logs are intentionally kept outside the repository.
