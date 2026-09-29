# A constrained haunted slab

This example sends a right-going **transverse** dark packet into a cold neutral electron–proton slab. It measures transmitted ordinary-photon energy through a vacuum detector plane. The result is a model conversion experiment, not a Galactic dark-matter detector prediction.

## Fixed physical experiment

We hold $\omega_0=10^9\,{\rm rad/s}$, $\mu=0.6\omega_0$, $\eta=0.05$, $\ell=c/\omega_0$, the $80\ell$ periodic box, and the total initial dark energy at $2\times10^{-5}\,{\rm J/m^2}$. The packet starts at $x=-16\ell$ with width $3.5\ell$; the photon detector is near $x=8\ell$. Its Fourier components satisfy $\omega_k^2=\mu^2+c^2\tilde k^2$, $\widehat E_{D,y}=i\operatorname{sgn}(k)\omega_k\widehat A_{D,y}$, and $B_D=\nabla_h\times A_D$. We normalize with the **full** dark electric, magnetic, and mass/potential energy. The packet is faster and broader than a nonrelativistic halo field; its role is to exercise propagation and conversion in the supported periodic solver.

Four smooth nonnegative basis densities $b_j$ have support $[-2\ell,2\ell]$, Gaussian width parameter $0.9\ell$, and a fixed smooth edge taper. The physical profile is

$$
n(x;\theta)=N_{\rm col}\sum_{j=1}^4 w_j(\theta)b_j(x),\quad
w=\operatorname{softmax}(\theta_1,\theta_2,\theta_3,0),\quad
N_{\rm col}=1.5n_{\rm ref}\ell,
$$

with $\theta_j\in[-0.6,0.6]$. Every candidate has the same column and support. The fine-grid peak stays below $n_{\rm ref}$ even at the eight corners of the enlarged $[-0.65,0.65]^3$ uncertainty box. Electron and proton particles are co-located with equal and opposite physical weights; changing a basis weight changes the species density, **not** charge-to-mass ratio. A cold profile at rest is an exact pressureless equilibrium up to deposition round-off. The no-wave control's largest ordinary electric field is **{{ profile_no_wave_E }} V/m**.

## Outgoing flux and independent reference

The PIC objective integrates the staggered ordinary Poynting flux $S_x=\epsilon_0c^2(E_yB_z-E_zB_y)$ at a fixed plane through $\omega_0t=60$, divided by the known incident dark energy. The scalar accumulator uses the production PIC transition and requests no output history; reverse-mode scan still keeps internal intermediates, as discussed in [performance](performance.md). The $\omega_0t=50$ to $60$ tail changes the held-out cold-design yield by **{{ profile_window_tail_percent }}%**. The significant packet and its reflected/transmitted waves cannot return to the detector through the periodic boundary in this window; the Gaussian's boundary tail is negligible at initialization. Open Proca boundaries are not implemented.

For each positive-$k$ incident packet mode, a separate sparse boundary-value solve uses

$$
\frac{d^2}{dx^2}\begin{pmatrix}A\\X\end{pmatrix}+
\left[\frac{\omega^2}{c^2}I-\frac1{c^2}
\begin{pmatrix}\omega_p^2&\eta\omega_p^2\\
\eta\omega_p^2&\mu^2+\eta^2\omega_p^2\end{pmatrix}\right]
\begin{pmatrix}A\\X\end{pmatrix}=0.
$$

Here $\omega_p^2$ includes **both** finite-mass species. Vacuum Robin conditions inject one dark mode from the left and allow outgoing photon/dark modes on both sides. The photon transmission is $(k_\gamma/k_D)|A_{\rm out}/X_{\rm in}|^2$, with $k_\gamma=\omega/c$ and $k_D=\sqrt{\omega^2-\mu^2}/c$. We average it with each packet's incident spectral energy weights. The maximum four-channel flux-sum error over reported scenarios is **{{ profile_reference_flux_error }}**. Separate tests check zero mixing, weak-mixing scaling, boundary-grid refinement, and an analytic uniform-slab transfer matrix. This cold frequency-domain solve is an **independent comparison**, not the objective optimized by JAX.

## Search and held-out replay

The deterministic training quadrature uses carrier $k_D\ell=0.75,0.85$ and coefficient shifts $\pm(0.03,-0.03,0.03)$. We maximize $\langle P_\gamma\rangle-10\operatorname{Var}(P_\gamma)-10^{-4}\sum_j(w_{j+1}-w_j)^2$ with three bounded gradient starts. Held-out carriers $0.70,0.90$ and larger shifts $\pm(0.05,-0.05,0.05)$ are never used to select the design.

| Training comparator | Objective |
|---|---:|
| Equal basis weights | {{ profile_uniform_objective }} |
| Single ramp | {{ profile_ramp_objective }} |
| Two-ramp | {{ profile_two_ramp_objective }} |
| Best equal-budget random search | {{ profile_random_objective }} |
| PIC gradient design | {{ profile_best_objective }} |

The coarse held-out photon fraction rises from **{{ profile_held_uniform }}** to **{{ profile_held_optimized }}**. With twice the cells, half the time step, and twice the particles per basis, the improvement is **{{ profile_refined_gain_min }}–{{ profile_refined_gain_max }}%** over three independent quantile phases. The cold spectral reference predicts the same ordering. The profile-direction centered finite difference and PIC reverse gradient differ by **{{ profile_gradient_error }}**. The gain is modest; the best of 24 random candidates is close, so we do not claim a unique or novel optimal structure.

![Fixed-column profiles and held-out outgoing photon fractions](_static/figures/profile_design/figure.png)

The [full run record](_static/figures/profile_design/run.json) contains every trial, failed start, random-search budget, uncertainty case, independent cold result, refined replay, and the [plot data](_static/figures/profile_design/data.npz). The quick preset is only a smoke run.

## Finite-amplitude kinetic replay

At $2000\,{\rm J/m^2}$ incident dark energy, the largest particle speed reaches **{{ profile_high_max_speed }}$c$**. The cold-design held-out photon fraction is **{{ profile_high_cold_yield }}**, versus **{{ profile_high_uniform_yield }}** for equal weights; nonlinear kinetic backreaction lowers its yield relative to the small-amplitude replay. At final time, the closed-energy drift is **{{ profile_high_energy_drift }}** of input, below one percent of the converted photon fraction. The cell-local random kinetic energy is **{{ profile_high_random_energy }}** of input, using one simulation cell as the coarse-graining scale; coherent cell-bulk motion is reported separately in the run record.

Re-optimizing at high amplitude gives **{{ profile_high_design_yield }}** on the coarse held-out set. Its extra gain over the cold design is comparable to numerical/loading variation after refined replay, so this run establishes **no reliable nonlinear optimization advantage**. The source of the yield change is self-consistent finite-amplitude particle response, but this single slab does not establish a universal depletion or saturation law. Broader amplitudes, independent ensembles, different profile families, and other detector models remain open.

Structured dark-photon conversion and optimized multilayer devices already exist in the literature: see [Baryakhtar, Huang and Lasenby (2018)](https://arxiv.org/abs/1803.11455) and the optimized experimental stack of [Manenti *et al.* (2021)](https://arxiv.org/abs/2110.10497). Inhomogeneous-plasma conversion was studied by [Aramburo Garcia *et al.* (2020)](https://arxiv.org/abs/2003.10465); [Hook, Huang and Shalaby (2025)](https://arxiv.org/abs/2510.13956) emphasize nonlinear plasma limits in a different, near-homogeneous resonant setting. Our result is a small, periodic 1D3V kinetic-model demonstration with a fixed-column slab and outgoing-flux measurement. It makes no priority or detector-sensitivity claim.
