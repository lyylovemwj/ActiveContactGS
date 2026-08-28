# Scientific scope and claim boundaries

ActiveContactGS is designed to test whether geometry-faithful Gaussian contact and active information-gain probing improve low-budget identification of latent contact structure and rigid-body parameters.

The repository supports three distinct evidence classes:

1. controlled procedural ID/OOD system-identification experiments;
2. contact-geometry evaluation on YCB object meshes;
3. native-protocol reproduction of an external Gaussian physical-world method.

These evidence classes must not be conflated. In particular:

- Gaussian-video experiments still use known Gaussian geometry tokens;
- YCB evaluates real object geometry, not real-robot system identification;
- native PIN-WM reproduction is not a fair common-task comparison by itself;
- internal Random, Fixed, isotropic, and no-shape controls are not substitutes for every public baseline;
- empirical gains do not guarantee universal SOTA status or conference acceptance.

Use `public_baseline_protocol.md` to define any public comparison before running it.
