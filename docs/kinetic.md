# A mixed kinetic whisper

The full longitudinal benchmark starts a small, fixed physical $k$ density perturbation in a current-neutral Maxwellian electron population with a uniform fixed neutralizing background. It uses 150,000 quiet particles, 64 cells, 1,200 steps, $k\lambda_D=0.5$, $\mu/\omega_p=1$, and $\eta=0.3$. The coupling is deliberately large enough for numerical validation; it is not an observational dark-matter parameter. The deposited **mean current remains physical** rather than being removed each step.

For $\sigma=\sqrt{T/m}$, $\zeta=\omega/(\sqrt2 k\sigma)$, and the Landau-continued plasma dispersion function $Z$, the independent reference solves

$$
\chi_L=\frac{\omega_p^2}{k^2\sigma^2}[1+\zeta Z(\zeta)],\qquad
D_L=(s-\mu^2)(1+\chi_L)+\eta^2s\chi_L=0,\qquad s=\omega^2-c^2k^2.
$$

The independent root has real and imaginary parts **{{ kinetic_root_real }}** and **{{ kinetic_root_imag }}** in $\omega_p$ units; the PIC fit gives **{{ kinetic_pic_real }}** and **{{ kinetic_pic_imag }}**. The real-frequency difference is **{{ kinetic_frequency_error }}%** and the damping-rate difference **{{ kinetic_damping_error }}%**. Five maxima between $\omega_pt=2.45$ and $11.225$ stand above the late mode floor; the fitted damping slope has a regression standard error of **{{ kinetic_damping_stderr }}** in $\omega_p$ units. The reference determinant residual and the exact window are stored in the [run record](_static/figures/mixed_kinetic/run.json).

![Seeded mixed kinetic mode and independent root envelope](_static/figures/mixed_kinetic/figure.png)

The late floor contains a coherent dark branch as well as particle noise, so an unqualified single-mode fit to the entire record would be wrong. The quick preset is explicitly noise-limited and makes no fit. This is **one resolved mixed Landau case**, not a convergence study of all kinetic roots. The full density/cell/time/seed refinements required for a kinetic convergence claim have not yet been completed.

The same full run now advances **JAX-in-Cell itself** from the identical electron positions, velocities, weights, ordinary field, grid and timestep. Setting $\eta=0$ in the independent determinant gives the ordinary [Landau dispersion](https://arxiv.org/abs/2303.12620) limit $\omega/\omega_p=1.41566-0.15336i$. The parent fit is $1.41195-0.15338i$; the dark run and mixed root above shift both frequency and damping. The [matched data](_static/figures/mixed_kinetic/data.npz) store all three field-mode histories. The displayed decay ends at the measured late floor; neither trace supports a single exponential fit through late times.

## Counterstreaming ordinary electrons

Two equal cold electron beams start at $\pm0.25c$ with equal physical weights, so their **deposited mean current is exactly zero**. A uniform fixed background neutralizes charge. We perturb one beam's position at a fixed $k$, leaving the mean current physical. For $a=(ku/\omega_p)^2$, $z=(\omega/\omega_p)^2$, $K=kc/\omega_p$ and $\widehat\mu=\mu/\omega_p$, the independent mixed cold determinant after removing beam poles is

$$
(z-K^2-\widehat\mu^2)[(z-a)^2-(z+a)]-\eta^2(z-K^2)(z+a)=0.
$$

The unstable root gives $\gamma/\omega_p=$ **{{ two_stream_root }}** at $\eta=0.3$; the full PIC fit over the **declared** $10<\omega_pt<20$ interval gives **{{ two_stream_pic }}** on 64 cells and **{{ two_stream_refined }}** on 128 cells with twice the particles and half the time step. The same loading at $\eta=0$ gives **{{ two_stream_zero_pic }}**. The observed mixing-induced growth-rate difference agrees in sign and scale with the determinant's $\eta=0$ to $0.3$ difference; the absolute growth error still includes cold-PIC discretization. The [full run record](_static/figures/mixed_two_stream/run.json) holds fit standard errors and both resolutions.

![Current-neutral mixed two-stream growth against a cold determinant](_static/figures/mixed_two_stream/figure.png)

### Through nonlinear saturation

`python examples/dark_saturation.py --full` follows the **same cold beam loading** in the parent and dark solvers through $\omega_pt=80$. Each starts with 64 cells and 4,000 particles per beam. One run halves $\Delta t$ at fixed grid/loading; another uses 128 cells and 8,000 particles per beam at the smaller step. The fitted parent/dark rates over $10<\omega_pt<20$ are $0.33228/0.34069$, $0.33407/0.34276$ and $0.33435/0.34302$. The independent cold roots are $0.33847/0.34670$. The growth-rate difference therefore survives these numerical refinements; absolute PIC rates are lower by roughly 1–2%.

![Matched two-stream growth, saturation, energy and phase space](_static/figures/two_stream_saturation/figure.png)

The first large mode peak lies near $\omega_pt\simeq29$. At that peak the simple trapping scale $\omega_b=\sqrt{2ek|E_{k_1}|/m_e}$ is about $1.54\gamma_{\rm parent}$ in the ordinary run and $1.47\gamma_{\rm mixed}$ in the dark run. The order-unity ratio and phase-space roll-up are **consistent with electron trapping**, the familiar nonlinear two-stream mechanism; this scale is not an exact saturation prediction. The [SHARP code paper](https://arxiv.org/abs/1702.04732) also emphasizes long-time energy control and joint particle/grid refinement for two-stream PIC.

The RMS of $|E_{k_1}|$ on $40\leq\omega_pt\leq80$ is $69.84/67.60$, $70.95/67.97$ and $68.62/64.56$ kV/m for parent/dark at the three settings. Thus the mixed case is 3.2–5.9% lower in this window. Late coherent phases and detailed vortex shapes vary with refinement, so a sharper saturation claim needs more particles and seed ensembles. The maximum relative total-energy changes are 0.70/0.68%, 0.70/0.68% and 0.23/0.19% for parent/dark. The **dark total includes the full Proca electric, magnetic and potential energy**; comparing ordinary-only energy in the mixed run would mislabel physical exchange as error. The maximum dark-field work residual, divided by the largest absolute dark source-work transfer, is $1.91\times10^{-5}$ at the base step and $4.65\times10^{-6}$ after halving it. The refined grid/loading case gives $6.91\times10^{-6}$. Dark work is therefore well tracked even where the full particle-plus-field energy drifts. Halving the step alone barely moves the latter error, while joint grid/loading refinement reduces it. This points to the particle/grid coupling as the dominant tested limitation; separating deposition order from particle noise needs further tests. The [run record](_static/figures/two_stream_saturation/run.json) and [arrays](_static/figures/two_stream_saturation/data.npz) retain all three cases. The source has no dark counterpart for separately dark-charged particles; both electron beams here carry ordinary charge and couple to the Proca field through $\eta$.

## Transverse anisotropy

A current-neutral, unmagnetized bi-Maxwellian has $\sigma_x=0.08c$ and $A=T_z/T_x=4$. We seed a transverse $E_z,B_y$ mode, then fit **the magnetic-mode magnitude** on $20<\omega_pt<70$; the electric mode has a large oscillatory transient. The independent Vlasov–Proca reference uses

$$
\Pi_T=\omega_p^2[1-A(1+\zeta Z(\zeta))],\quad
\zeta=\frac{\omega}{\sqrt2k\sigma_x},\quad
(s-\mu^2)(s-\Pi_T)-\eta^2s\Pi_T=0.
$$

Its growing root is **{{ weibel_root }}** $\omega_p$; PIC gives **{{ weibel_pic }}** at 64 cells/30,000 particles and **{{ weibel_refined }}** at 128 cells/60,000 particles. The $\eta=0$ control is **{{ weibel_zero_pic }}**. The individual mixed-root fit agrees closely, but the predicted **difference** from zero coupling is comparable to fit uncertainty and is **not yet resolved as a mixing effect**. This is one anisotropy benchmark, not a survey of unstable branches. The [run record](_static/figures/mixed_weibel/run.json) includes the fit windows and uncertainty.

![Current-neutral transverse anisotropy magnetic-mode growth](_static/figures/mixed_weibel/figure.png)

These two-stream and Weibel cases involve **ordinary charged species responding to both fields**. A theory in which particles carry a separate dark charge and interact through a massive mediator alone is a different model and is not implemented here. All $\eta=0.3$ cases use a deliberately large coupling to resolve numerical effects, with no observational interpretation. The quick presets are smoke tests without fitted growth rates.

## Mobile ions and a finite reservoir

The [mobile-ion example](../examples/dark_reservoir.py) uses co-located, exactly charge-neutral electron and proton loadings with $m_i/m_e$ at its physical value, $T_e=T_i=10^{-3}m_ec^2$, and a fixed seeded velocity mode. It compares zero drive, a prescribed longitudinal $F\cos\omega_pt$, and two dynamical Proca fields whose **initial** effective force $\eta E_D$ is the same $F=0.03m_e\omega_pv_{\rm th,e}/e$. Both Proca rest frequencies equal $\omega_p$. The small and large reservoirs set $\eta=0.2$ and $0.02$, giving initial dark energy **{{ mobile_small_energy_ratio }}** and **{{ mobile_large_energy_ratio }}** times the particles' initial longitudinal kinetic energy. Changing $\eta$ changes the reservoir size and backreaction while holding the initial force fixed.

An independent homogeneous two-fluid system evolves the mean ordinary field, dark field and both species velocities. It contains the finite ion mass and the same $\eta J$ source, but no PIC deposition. With 64 cells/4,000 particles per species and then 128 cells/8,000 particles per species at half the step, the refined maximum mean-field error over $F$ is **{{ mobile_external_oracle_error }}** for the external drive and **{{ mobile_large_oracle_error }}** for the large reservoir. The [run record](_static/figures/mobile_ions/run.json) gives both resolutions, all four cases and initial/final energy components. Over $\omega_pt\leq5$, the large reservoir's effective pump differs from the imposed cosine by at most **{{ mobile_large_early_pump_error }}$F$**. Through $\omega_pt=40$, the small reservoir gives up **{{ mobile_small_depletion }}%** of its initial dark energy and the large one **{{ mobile_large_depletion }}%**. The maximum absolute closed/work balance error over the declared initial-energy scale is **{{ mobile_max_balance }}**. The one-cell, species-specific random longitudinal kinetic measure changes by less than one percent in all cases; this run does not establish nonlinear heating or exponential growth.

![Matched mobile-ion prescribed and finite-reservoir controls](_static/figures/mobile_ions/figure.png)

The [reported SHARP setup, Appendix B](https://arxiv.org/html/2510.13956v1) uses $L=40c/\omega_p$, 1,000 cells ($\Delta x=0.04c/\omega_p$), fifth-order particle shapes and a much longer horizon. Its Eq. (45), with $v_q^D/v_{\rm th,e}=0.03$, $v_{\rm th,e}/c=\sqrt{10^{-3}}$, and quoted noise-onset target $\omega_pt\simeq2.4$, implies roughly $2.06\times10^5$ total macroparticles **if those quantities are inserted literally**; the paper does not give that as an explicit per-species loading count. Our refined control uses 128 cells, $L=2\pi c/\omega_p$, 8,000 particles per species and $\omega_pt\leq40$. The box, shape order, loading and horizon differences preclude a claim that its later heating or saturation curves have been reproduced.

### Early paper-geometry pilot

`python examples/dark_reservoir.py --paper-pilot --output artifacts/paper_pilot` runs the stronger prescribed force and a matched zero-drive control on the reported 1,000-cell, $40c/\omega_p$ grid through $\omega_pt=80$. It repeats both with 20,000 and 40,000 quiet particles **per species**, seeded with the same $0.01v_{\rm th,e}$ velocity mode. The drive and mobile-ion mean field follow the independent two-fluid solution to within $0.031F$; the largest energy/work balance error is $0.0026$ of the initial particle energy. At 20,000 particles per species, the peak nonzero-$k$ electric energy is $0.0161$ of initial particle energy without the drive and $0.0166$ with it. At 40,000, those fractions fall to $0.00472$ and $0.00444$. The drive-minus-zero difference changes sign under loading refinement, so this pilot resolves **no pump-induced higher-mode growth** by $\omega_pt=80$. The cell-local random kinetic measure falls slightly in both controls; it gives no heating evidence here.

![Early strong-drive response and loading-sensitive nonzero-mode energy](_static/figures/paper_geometry_pilot/figure.png)

The [run record](_static/figures/paper_geometry_pilot/run.json) and [plotted arrays](_static/figures/paper_geometry_pilot/data.npz) retain both loadings. This is an early-time diagnostic with the parent's quadratic particle shape, $\leq80{,}000$ total particles and a horizon far short of the source's long runs. It neither reproduces nor rules out the reported nonlinear heating and saturation.
