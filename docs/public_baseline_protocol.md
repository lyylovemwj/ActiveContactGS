# Public Baseline Protocol and Availability Audit

> Audit date: 2026-08-26  
> Purpose: define comparisons that are reproducible and faithful to each public method, without silently adding capabilities that the original method does not support.

## Availability

| Method | Official source | Code status on audit date | Planned use |
|---|---|---|---|
| PIN-WM | [Official repository](https://github.com/XuAdventurer/PIN-WM), commit `99d3fde5d233aeffabfa287f94831cf7c7afee64` | Released: data collection, 2DGS training, physical parameter identification | Run the official native Push-T pipeline and a supported common-subset identification protocol |
| ContactGaussian-WM | [Official project page](https://contactgaussian-wm.github.io/), [arXiv](https://arxiv.org/abs/2602.11021) | The project page describes the method but exposes no downloadable implementation; its displayed “Code” item does not resolve to a code repository | Paper-reported comparison and geometry/contact protocol only; no fabricated reimplementation will be called “official” |

## Comparison is split into two questions

### A. World-model and identification quality

This table compares capabilities shared with passive differentiable system-identification methods.

Common fields:

- same object/scene subset when the public implementation supports it;
- same ground-truth physical parameter draw;
- same initial state and applied intervention trajectory;
- same number of observed trajectories;
- same rollout horizon;
- parameter error and predictive rollout error;
- wall-clock optimization/inference time after explicit warm-up.

Methods do not have to infer the same latent structure. A method that assumes a fixed collider is evaluated under that assumption and marked accordingly.

### B. Active identification quality

This table is restricted to methods that select interventions. PIN-WM and ContactGaussian-WM are not retrofitted with an invented active selector and are therefore marked “not supported” for Active EIG.

Compared active selectors:

- Counterfactual joint EIG (ours);
- covariance information score;
- predictive variance;
- fixed diverse probes;
- random probes;
- high-energy random probes.

Metrics:

- normalized physical-parameter error;
- discrete contact-structure accuracy where applicable;
- predictive rollout RMSE;
- continuous downstream control error;
- thresholded control success as a secondary metric;
- FIM log-determinant.

## PIN-WM execution levels

Results must state which level was actually executed:

1. **Official/native**: unmodified official Push-T scene, data collection, 2DGS training, differentiable simulation and rendering loss.
2. **Official-core/common subset**: official differentiable physics and optimizer, with only dataset/config adapters required to consume the common scene.
3. **PIN-WM-style proxy**: our simulator with per-instance gradient optimization. This may be informative but must never be labeled an official PIN-WM result.

Primary paper tables should prefer levels 1 or 2. Level 3 belongs in an appendix and must be visibly labeled.

## ContactGaussian-WM policy

Because no runnable official implementation was found on the audit date:

- compare the paper's stated task, assumptions and reported numbers where metrics are directly compatible;
- use our YCB contact benchmark to isolate the spherical-Gaussian collision assumption, but call it a **representation ablation**, not an execution of ContactGaussian-WM;
- do not claim that the original method obtains our measured sphere-baseline numbers unless its released code and exact collision rule become available;
- record the project-page and arXiv versions used for the audit.

## Required reporting metadata

Every public-baseline row must include or link to:

- repository URL and exact commit;
- environment lock file;
- upstream files modified, with a patch;
- raw command line;
- input data manifest;
- number of optimization steps/observations;
- GPU model, peak memory and timing protocol;
- raw per-task output and aggregation script;
- whether the result is official/native, official-core, or proxy.

## Fairness constraints

- Do not give our method fewer observations only to make the public method appear weak.
- Do not count offline amortized training as per-task inference time; report offline training separately.
- Do report PIN-WM per-scene optimization time and our amortized inference time because they are genuinely different inference paradigms.
- Do not compare rendering PSNR to physical rollout RMSE as if they were the same metric.
- Do not tune a control threshold separately per method or per distribution.
- Failed environment builds and unsupported tasks are documented as availability facts, not converted into numerical wins.
