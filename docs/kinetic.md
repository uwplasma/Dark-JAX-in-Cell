# A mixed kinetic whisper

The full longitudinal benchmark starts a small, fixed physical $k$ density perturbation in a current-neutral Maxwellian electron population with a uniform fixed neutralizing background. It uses 150,000 quiet particles, 64 cells, 1,200 steps, $k\lambda_D=0.5$, $\mu/\omega_p=1$, and $\eta=0.3$. The coupling is deliberately large enough for numerical validation; it is not an observational dark-matter parameter. The deposited **mean current remains physical** rather than being removed each step.

For $\sigma=\sqrt{T/m}$, $\zeta=\omega/(\sqrt2 k\sigma)$, and the Landau-continued plasma dispersion function $Z$, the independent reference solves

$$
\chi_L=\frac{\omega_p^2}{k^2\sigma^2}[1+\zeta Z(\zeta)],\qquad
D_L=(s-\mu^2)(1+\chi_L)+\eta^2s\chi_L=0,\qquad s=\omega^2-c^2k^2.
$$

The reference root is $({{ kinetic_root_real }}{{ kinetic_root_imag }}i)\omega_p$; the PIC mode fit is $({{ kinetic_pic_real }}{{ kinetic_pic_imag }}i)\omega_p$. The real-frequency difference is **{{ kinetic_frequency_error }}%** and the damping-rate difference **{{ kinetic_damping_error }}%**. Five maxima between $\omega_pt=2.45$ and $11.225$ stand above the late mode floor; the fitted damping slope has a regression standard error of **{{ kinetic_damping_stderr }}** in $\omega_p$ units. The reference determinant residual and the exact window are stored in the [run record](_static/figures/mixed_kinetic/run.json).

![Seeded mixed kinetic mode and independent root envelope](_static/figures/mixed_kinetic/figure.png)

The late floor contains a coherent dark branch as well as particle noise, so an unqualified single-mode fit to the entire record would be wrong. The quick preset is explicitly noise-limited and makes no fit. This is **one resolved mixed Landau case**, not a convergence study of all kinetic roots. The full density/cell/time/seed refinements required for a kinetic convergence claim have not yet been completed.

## Counterstreaming ordinary electrons

Two equal cold electron beams start at $\pm0.25c$ with equal physical weights, so their **deposited mean current is exactly zero**. A uniform fixed background neutralizes charge. We perturb one beam's position at a fixed $k$, leaving the mean current physical. For $a=(ku/\omega_p)^2$, $z=(\omega/\omega_p)^2$, $K=kc/\omega_p$ and $\widehat\mu=\mu/\omega_p$, the independent mixed cold determinant after removing beam poles is

$$
(z-K^2-\widehat\mu^2)[(z-a)^2-(z+a)]-\eta^2(z-K^2)(z+a)=0.
$$

The unstable root gives $\gamma/\omega_p=$ **{{ two_stream_root }}** at $\eta=0.3$; the full PIC fit over the **declared** $10<\omega_pt<20$ interval gives **{{ two_stream_pic }}** on 64 cells and **{{ two_stream_refined }}** on 128 cells with twice the particles and half the time step. The same loading at $\eta=0$ gives **{{ two_stream_zero_pic }}**. The observed mixing-induced growth-rate difference agrees in sign and scale with the determinant's $\eta=0$ to $0.3$ difference; the absolute growth error still includes cold-PIC discretization. The [full run record](_static/figures/mixed_two_stream/run.json) holds fit standard errors and both resolutions.

![Current-neutral mixed two-stream growth against a cold determinant](_static/figures/mixed_two_stream/figure.png)

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
