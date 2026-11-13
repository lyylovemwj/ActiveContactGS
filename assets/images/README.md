# Manuscript figure provenance

The PNG files in this directory are high-resolution extractions from the
authors' supplied manuscript PDFs. The PDFs themselves are not redistributed.

## Sixteen-page ICLR manuscript

Source snapshot:

- local source filename: `__Active_PhysicalIdentification_with_Gaussian_Geometry__2_ (8).pdf`;
- source SHA-256: `1f17f3a84f65f3301cd6c4a9a7c3ff21413c4ba8e7474ed40c2e54c06508e42d`;
- extraction date: 2026-11-13;
- extraction: 4x PDF rendering (approximately 288 DPI), cropped to the published figure/table bounds.

| Repository file | Manuscript mapping | Description |
|---|---|---|
| `iclr/fig1_factorized_overview.png` | Figure 1, page 2 | Factorized posterior, tuple recombination, and Active-EIG loop |
| `iclr/fig2_physics101_objects.png` | Figure 2, page 6 | Representative Physics101 real objects |
| `iclr/fig3_probe_budget.png` | Figure 3, page 6 | Exact-world accuracy versus probe budget |
| `iclr/fig4_physics101_low_label.png` | Figure 4, page 7 | Corrected Physics101 low-label sweep |
| `iclr/fig5_real_contact_dataset.png` | Figure 5, page 8 | Self-collected cube, sphere, and box push sequences |
| `iclr/fig6_eig_calibration.png` | Figure 6, page 9 | Predicted and realized information gain |
| `iclr/table3_formal_evaluation.png` | Table 3, page 7 | Formal six-probe evaluation |
| `iclr/table4_factorization_ablation.png` | Table 4, page 7 | Factorized versus flat inference |
| `iclr/table5_real_world_summary.png` | Table 5, page 8 | Real-world evidence summary |

## Earlier eight-page manuscript

The files directly under `assets/images/` preserve the earlier 3DV snapshot.

Source snapshot:

- local source filename: `ActiveContactGS(1).pdf`;
- source SHA-256: `ab969e7ba084cfb65dc732db5d76f31e5b8488074c9fe73b352aaac523e8b8ea`;
- extraction date: 2026-08-28.

| Repository file | Manuscript mapping | Description |
|---|---|---|
| `fig1_method_overview.png` | Figure 1, page 2 | ActiveContactGS architecture and active belief-update loop |
| `fig3_sample_efficiency.png` | Figure 3, page 4 | Probe-budget curves and contact-normal fidelity |
| `fig4_geometry_ablation.png` | Figure 4, page 5 | Geometry and action-selection ablation |
| `fig5_anisotropic_contact.png` | Figure 5, page 6 | Native anisotropic contact versus a sphere approximation |
| `fig7_cross_object.png` | Figure 7, page 7 | Cross-object active probing on cup, bunny, and Venus |

These are original ActiveContactGS manuscript figures. When the final archival paper changes, re-extract the images and update both this mapping and the source checksum.
