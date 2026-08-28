# ActiveContactGS: working project specification

## Research question

Can an object-centric Gaussian digital twin disambiguate multiple physically
different worlds that explain the same passive video by choosing a safe
counterfactual contact and updating a calibrated multi-hypothesis belief?

## Core hypothesis

Under an equal real-interaction budget, counterfactual-disagreement probing reduces
joint geometry/physical-parameter error and long-horizon prediction error, and
raises belief-aware MPC success on unseen objects.

## Proposed system

1. Reconstruct canonical object-centric anisotropic Gaussian assets and use the
   native ellipsoids as differentiable collision geometry without sphere forcing.
2. Maintain a mixture posterior over discrete contact modes/collider hypotheses
   and continuous mass, inertia, friction, and restitution.
3. Render/simulate candidate pushes through a differentiable contact model.
4. Score candidates by expected posterior entropy reduction in Gaussian image
   space, subject to contact-force and workspace safety constraints.
5. Execute one probe, compare predicted and observed multi-view evidence, update
   the posterior, and repeat.
6. Plan the task with belief-space or risk-sensitive MPC.

## Bottom-layer geometry claim

ContactGaussian-WM forces isotropic spherical splats for closed-form collision.
Its paper reports reduced rendering quality and inaccurate penetration/contact
normals under deep overlap. We instead optimize the support-plane separation of
two oriented Gaussian ellipsoids with fixed unrolled Riemannian steps, producing
a signed gap, opposing support points, and a differentiable contact normal. The
claim must be validated against exact mesh/convex collision queries, sphere-union
and extracted-mesh baselines, finite-difference gradients, and rendering quality.

## Required comparisons

- Fixed scripted probes.
- Random probes with the same action bounds and budget.
- ASID-style active system identification from privileged state.
- PIN-WM-style passive/random few-shot identification.
- ContactGaussian-WM deterministic point estimation.
- Oracle geometry and oracle physical-parameter upper bounds.

## Metrics

- Relative error for mass, friction, restitution, and principal inertia.
- Posterior negative log likelihood, coverage, and expected calibration error.
- Translation/rotation trajectory RMSE at 1, 2, and 5 seconds.
- Contact timing and contact-pair precision/recall.
- Downstream MPC success and collision/penetration violations.
- Number of real probes needed to hit fixed error/success thresholds.
- PSNR, SSIM, and LPIPS only as observation/rendering diagnostics.

## Go/no-go gates

- G1: active probing beats random by at least 20% median normalized parameter
  error after 3 probes on held-out synthetic objects.
- G2: advantage persists with image noise, pose uncertainty, and collider mismatch.
- G3: at least 3 real objects show the same direction of improvement.
- G4: Gaussian image-space likelihood adds measurable value beyond privileged pose.

Failing G1 means the paper premise is rejected or redesigned before expensive 3DGS
integration. Failing G2 means geometry/observation uncertainty must be modeled more
carefully. G3 and G4 are required for a competitive full paper.
