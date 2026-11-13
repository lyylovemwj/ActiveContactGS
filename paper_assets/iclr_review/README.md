# ICLR reviewer asset index

This directory contains paper-ready figures, tables, and machine-readable
sources generated from `outputs/iclr-review-formal`. PNG files support quick
inspection; PDF files are vector versions for LaTeX. Assets are evidence for
specific claims, not decorative substitutes for quantitative results.

## Recommended placement

| Asset | Visual grammar | Claim supported | Source data | Placement |
|---|---|---|---|---|
| `representation_comparison` | annotated heatmap | factorization enables unseen-composition generalization | `representation_policy_results.csv` | main text |
| `coverage_controls` | grouped bars and intervals | unrestricted Active EIG beats constrained and fixed policies | `representation_policy_results.csv` | main text |
| `action_selection_fingerprints` | polar rose plots | policies have distinct directional probing signatures | `action_selection_results.csv` | main/qualitative |
| `per_task_distributions` | violin and box plots | means hide task-level recovery failures | `per_task_results.csv` | appendix |
| `uncertainty_failure_landscape` | hexbin density maps | uncertainty, error, and failures are coupled nonlinearly | `per_task_results.csv` | appendix |
| `physical_response_atlas` | spatial trajectory small multiples | latent modes create different responses under matched pushes | `physical_response_trajectories.csv` | method/appendix |
| `model_mismatch` | robustness curves | tested mismatch does not cause monotonic accuracy degradation | `model_mismatch_results.csv` | response/appendix |
| `eig_sample_sensitivity` | variance, agreement, and runtime curves | variance falls while top-1 agreement stays weak | `eig_sample_results.csv` | limitations/appendix |
| `identifiability_boundary` | boundary curves | near-isotropy causes an observation-equivalence collapse | `identifiability_results.csv` | main/limitations |

`main_results_table.tex` summarizes Active-EIG exact accuracy for the four
posterior representations and three evaluation splits.

## Reading rules

- The heatmap compares representations and splits; it does not establish cause.
- Polar radial values are action frequencies, not accuracy.
- Violin tails and multimodality are part of the result, not noise to hide.
- Sparse hexbins contain few tasks and should not be overinterpreted.
- The trajectory atlas is illustrative; formal claims come from evaluation CSVs.
- Do not describe the mismatch curve as monotonic.
- Do not claim EIG action convergence: 64-sample top-1 agreement is 39.1%.

## Reproduction

From the repository root on a machine with Matplotlib installed:

```bash
python -m active_contact_gs.plot_iclr_review_assets \
  --results outputs/iclr-review-formal \
  --output paper_assets/iclr_review
```

The generator writes PNG, PDF, CSV, TeX, and manifest artifacts. It leaves this
README in place and records it separately as documentation in `manifest.json`.
