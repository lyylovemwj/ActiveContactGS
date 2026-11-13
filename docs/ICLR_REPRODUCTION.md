# ICLR reconstruction and reviewer experiment suite

This document describes the independent code reconstruction of the 16-page
ActiveContactGS ICLR manuscript and the additional reviewer experiments. It is
not an upstream author release and should not be presented as one. The code
follows the manuscript's stated problem structure and records the assumptions
needed to make the protocol executable.

## Implemented scope

- 243 discrete physical worlds from five three-level factors.
- 144 candidate contact actions.
- Flat 243-way, independent, autoregressive, and structured-joint posteriors.
- Active expected information gain (EIG), direction-matched EIG, balanced
  fixed, random, and fixed probing policies.
- Held-out ID, composition-OOD, and geometry-OOD splits.
- Model-mismatch, EIG sampling, and near-isotropy diagnostics.
- Per-task outputs and source CSV files for every paper figure.

## Formal protocol

The completed RTX 4090 run used these frozen settings:

| Item | Value |
|---|---:|
| Training tasks | 262,144 |
| Tasks per validation/test split | 16,384 |
| Posterior representations | 4 |
| Seeds per representation | 3 (7301, 7302, 7303) |
| Main training runs | 12 |
| Candidate actions | 144 |
| Evaluation tasks per run and split | 128 |

Run the complete protocol from the repository root:

```bash
export ACGS_ICLR_DATA_ROOT=data/iclr-v1
export ACGS_ICLR_OUTPUT_ROOT=outputs/iclr-review-formal
bash scripts/run_iclr_reviewer_suite.sh
```

The script generates data, trains all models, evaluates every policy, runs the
diagnostics, and builds the assets. Individual CLI arguments remain the source
of truth.

## Main results

Active-EIG exact-world accuracy (mean +/- sample standard deviation over three
seeds) is also exported as `paper_assets/iclr_review/main_results_table.tex`:

| Representation | ID | Composition OOD | Geometry OOD |
|---|---:|---:|---:|
| Flat 243-way | 87.0 +/- 2.4 | 0.0 +/- 0.0 | 89.6 +/- 2.0 |
| Independent | 83.6 +/- 5.1 | 75.3 +/- 6.5 | 79.4 +/- 5.9 |
| Autoregressive | **94.3 +/- 0.9** | 81.8 +/- 2.4 | **94.3 +/- 0.9** |
| Structured joint | 89.8 +/- 1.6 | **83.6 +/- 4.1** | 93.0 +/- 1.4 |

The flat classifier reaches exactly zero composition-OOD exact accuracy because
an unseen tuple has no trained class support. Factorized representations recover
substantial composition-OOD accuracy. Autoregressive is strongest on ID and
geometry OOD; structured joint is strongest on composition OOD.

Across every representation and split, unrestricted Active EIG beats
direction-matched EIG, balanced fixed, random, and fixed probing on exact-world
accuracy. Comparisons are paired by representation, seed, split, and task in
`representation_policy_results.csv`.

## Reviewer diagnostics

### Model mismatch

For independent seed 7301, Active-EIG exact accuracy is 78.9%, 76.6%, 78.1%,
76.6%, and 82.0% at mismatch levels 0%, 5%, 10%, 20%, and 30%. This is
non-monotonic and approximately stable over the tested range; it does not
support a claim of graceful monotonic degradation. Parameter MAE increases from
0.0093 at 0% to 0.0140 at 30%.

### Near-isotropy and identifiability

Exact-world accuracy falls from 82.0% at aspect ratio 0.7 to 49.2% at 0.9,
21.1% at 0.95, and 22.7% at 1.0. Native and rounded trajectory MSE becomes
exactly zero at 1.0, making those modes observationally indistinguishable in the
simulator. This is an identifiability boundary, not an optimizer failure.

### EIG Monte Carlo sensitivity

Mean score variance decreases from 1.816 with 2 samples to 0.048 with 64.
However, top-1 action agreement with the 64-sample reference is only 39.1% even
at 64 samples across repeated acquisitions. The action ranking is therefore not
converged. Runtime increases from 1.03 to 1.46 seconds per acquisition.

## Paper assets

Different questions use different visual encodings:

| Asset | Visual form | Purpose |
|---|---|---|
| `representation_comparison` | annotated heatmap | representation generalization |
| `coverage_controls` | grouped bars with intervals | probing-policy comparison |
| `action_selection_fingerprints` | polar rose plots | directional action preferences |
| `per_task_distributions` | violin and box plots | task-level recovery distributions |
| `uncertainty_failure_landscape` | hexbin density maps | uncertainty-error relationships |
| `physical_response_atlas` | spatial trajectories | simulator response mechanisms |
| `model_mismatch` | robustness curves | misspecification sensitivity |
| `eig_sample_sensitivity` | diagnostic curves | variance, ranking, and cost trade-offs |
| `identifiability_boundary` | boundary curves | near-isotropic failure regime |

Each figure has PNG and vector PDF versions plus a machine-readable CSV source.
The claim-to-asset map is in `paper_assets/iclr_review/README.md`, and
`manifest.json` is the authoritative generated-file inventory.

Regenerate assets from completed results with:

```bash
python -m active_contact_gs.plot_iclr_review_assets \
  --results outputs/iclr-review-formal \
  --output paper_assets/iclr_review
```

## Artifact locations

- Lightweight local results: `outputs/iclr-review-formal/`
- Local assets: `paper_assets/iclr_review/`
- Formal log: `outputs/iclr-review-formal.log`
- Full checkpoints remain in external run storage and are not part of the
  lightweight repository.

Checkpoints and tensor shards are intentionally outside the lightweight
repository. Credentials, reviewer identities, and private server details must
never be written into release artifacts.

## Interpretation limits

- These results come from an independent reconstruction, not original author
  checkpoints.
- Evaluation uses 128 tasks per run and split; intervals are not population
  guarantees.
- Composition OOD is a strict unseen-tuple split and especially disadvantages
  a closed-set flat classifier.
- Stable downstream accuracy does not imply stable EIG action rankings.
- The mismatch sweep covers only the implemented perturbation family and range.
- The physical-response atlas is illustrative; quantitative claims must use the
  CSV-backed formal experiments.
