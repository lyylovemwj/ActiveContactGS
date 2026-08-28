# Research landscape (living document, checked 2026-08-26)

## Novelty correction

The initially proposed "object Gaussians + contact graph + differentiable rigid
integrator" is already substantially occupied:

| Work | What it already covers | Remaining gap relevant here |
|---|---|---|
| [PIN-WM (RSS 2025)](https://arxiv.org/abs/2504.16693) | Differentiable rigid system identification from Gaussian-rendered observations; few-shot random interactions; digital cousins; Sim2Real | No adaptive next-contact selection or calibrated physical posterior |
| [ContactGaussian-WM (2026)](https://arxiv.org/abs/2602.11021) | Gaussian appearance/collision geometry, differentiable collision detection and closed-form contact dynamics, sparse video fitting, MPC | Deterministic/passive identification; project code was not linked as released when checked |
| [MRO-GWM (2026)](https://arxiv.org/abs/2606.01950) | Object-centric Gaussian assets, action-conditioned multi-rigid dynamics, occlusion and MPC | Learned pose dynamics rather than active physical identifiability |
| [PersistGS (CVPRW 2026)](https://openaccess.thecvf.com/content/CVPR2026W/GenRecon3d/html/Ramlal_PersistGS_Differentiable_Physics_for_Object_Permanence_in_4D_Gaussian_Splatting_CVPRW_2026_paper.html) | Differentiable physics through complete occlusion; estimates friction and velocity | Three synthetic single-ball scenes; no active probing or multi-parameter belief |
| [PhysGS (2025)](https://arxiv.org/abs/2511.18570) | Bayesian dense physical properties and uncertainty from visual/VLM cues | Static semantic/material inference, not dynamic contact system identification |
| [ASID (ICLR 2024 Oral)](https://arxiv.org/abs/2404.12308) | Fisher-information exploration policies for simulator refinement; identifies friction/inertia and enables Sim2Real | Assumes a prebuilt simulator/state pipeline; no Gaussian visual likelihood, discrete contact-mode belief, or joint geometry/physics uncertainty |
| [Physically Embodied GS (2024)](https://arxiv.org/abs/2406.10788) | Online visually correctable particle/Gaussian world at 30 Hz | Known physics; does not actively identify hidden properties |
| [PhysGaussian (CVPR 2024)](https://openaccess.thecvf.com/content/CVPR2024/html/Xie_PhysGaussian_Physics-Integrated_3D_Gaussians_for_Generative_Dynamics_CVPR_2024_paper.html) | Physics-integrated Gaussian simulation using MPM | Generative continuum dynamics, not active rigid system ID |

The defendable contribution is therefore not generic active system identification.
The sharper target is a *multi-hypothesis visual contact belief*: preserve discrete
contact-mode and continuous geometry/physics alternatives that explain the same
passive video, select a safe counterfactual contact whose Gaussian renderings most
disagree, and update from image evidence. This directly targets physical
identifiability under occlusion and geometry/parameter confounding.

## Submission timing

[3DV 2027](https://3dvconf.github.io/2027/call-for-papers/) lists abstract
registration on 2026-08-21 and the paper deadline on 2026-08-28 at 11:00 PDT.
Starting a new empirical project on 2026-08-26 cannot responsibly meet that cycle
unless a submission was already registered and substantial work already exists.
