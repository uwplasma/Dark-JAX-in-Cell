# Maxwell–Proca equations

## Canonical fields and SI units

The ordinary potential $A_\mu$ and dark potential $X_\mu$ have diagonal kinetic terms. We define $\eta$ as the **exact** ratio of dark to ordinary current coupling. For an original mixing term $+\epsilon\widetilde F\widetilde G/2$ and bare mass $m_0$,

$$
\widetilde A=A+\epsilon\widetilde X,\quad
X=\sqrt{1-\epsilon^2}\,\widetilde X,\quad
\eta=\epsilon/\sqrt{1-\epsilon^2},\quad
\mu=m_0/\sqrt{1-\epsilon^2}.
$$

Substituting the first relation cancels the kinetic cross term; rescaling $X$ makes its kinetic term canonical. The input `omega` is the physical rest frequency $\Omega_D=\mu c^2/\hbar$ in rad/s. We do not add a plasma mass to the field equation: the particles generate plasma response.

The dark electric field $\mathbf E_D$ is in V/m, magnetic field $\mathbf B_D$ in T, vector potential $\mathbf A_D$ in V s/m, scalar potential $\phi_D$ in V, and charge/current densities in C/m³ and A/m². The implemented SI equations are

$$
\begin{aligned}
\dot{\mathbf E}_D &= c^2\nabla\times\mathbf B_D+\Omega_D^2\mathbf A_D-\eta\mathbf J/\epsilon_0,\\
\dot{\mathbf B}_D &= -\nabla\times\mathbf E_D,\\
\dot{\mathbf A}_D &= -\mathbf E_D-\nabla\phi_D,\\
\dot\phi_D &= -c^2\nabla\cdot\mathbf A_D,\\
\nabla\cdot\mathbf E_D+\Omega_D^2\phi_D/c^2 &= \eta\rho_{\rm total}/\epsilon_0.
\end{aligned}
$$

The ordinary Maxwell field obeys $\dot{\mathbf E}=c^2\nabla\times\mathbf B-\mathbf J/\epsilon_0$ and $\dot{\mathbf B}=-\nabla\times\mathbf E$. Its Gauss law uses the same total charge. For periodic electron-only loading, $\rho_{\rm total}=\rho_{\rm particles}-\langle\rho_{\rm particles}(0)\rangle$: the fixed neutralizing background carries no current. The **physical mean current stays in Ampère's law**. Particles feel $q[\mathbf E+\eta\mathbf E_D+\mathbf v\times(\mathbf B+\eta\mathbf B_D)]$ using the parent's Newtonian or relativistic Boris pusher. Statistical weights multiply sources and energy, not $q/m$.

## Compatible update

Charge and $\phi_D$ live at cell centres; $\mathbf E_D$ and $\mathbf A_D$ live at electric faces; $\mathbf B_D$ lives at magnetic centres. Let $D$ be the face-to-centre divergence and $G=-D^*$ the centre-to-face gradient. The parent compatible curls give $D\operatorname{curl}=0$ and $\operatorname{curl}G=0$.

For a half-step $h$, the kick changes $\mathbf E_D$ by $h(c^2\operatorname{curl}\mathbf B_D+\Omega_D^2\mathbf A_D-\eta\mathbf J/\epsilon_0)$ and $\phi_D$ by $-hc^2D\mathbf A_D$. The drift changes $\mathbf A_D$ by $-h(\mathbf E_D+G\phi_D)$ and $\mathbf B_D$ by $-h\operatorname{curl}\mathbf E_D$. We use kick–drift before the particle push and drift–kick after it, with JAX-in-Cell's two trajectory-current deposits. In the kick, the mass terms cancel in $D\mathbf E_D+\Omega_D^2\phi_D/c^2$; continuity supplies the remaining charge change. The drift preserves both Gauss and $\mathbf B_D-\operatorname{curl}\mathbf A_D$.

Initialization sets the longitudinal electric fields from both Gauss laws. A supplied `DarkField.initial_E[:, 0]` is **projected** onto the dark Gauss solution; its spatial mean is retained and the L2 size of the correction is returned as `state.initial_projection_norm`. `project_initial_electric` exposes the same operation for preparation and inspection. The default `E_D = eta E`, `phi_D = 0` is a compatible bare-field preparation, not a screened equilibrium or a single in-medium eigenmode. The parent's quadratic particle shape, charge-conserving current and physical mean-current closure feed both field updates; matching divergence and curl differences then propagate the constraints. The dark extension uses the parent's explicit electromagnetic Ampère solver with collisionless particles and unsmoothed sources in a periodic box.

For source-free fields the mass adds a frequency even in a homogeneous box. The field-only stability bound is $\Delta t\sqrt{4c^2/\Delta x^2+\Omega_D^2}<2$; construction uses a 1.9 margin. Resolve plasma and gyro frequencies separately. No implicit or SOLVAX solve is used in this explicit step.

## Energy and work

The full closed energy per transverse area is

$$
U=\sum_p w_p K_p+\frac{\epsilon_0}{2}\int dx\,
\left[|\mathbf E|^2+c^2|\mathbf B|^2+|\mathbf E_D|^2+c^2|\mathbf B_D|^2
+\Omega_D^2(|\mathbf A_D|^2+\phi_D^2/c^2)\right].
$$

The tracked dark source work approximates $-\eta\int dt\,dx\,\mathbf J\cdot\mathbf E_D$ with fields centred on each source kick. A prescribed drive takes an explicit three-component electric-amplitude vector and accumulates external particle work from the Boris mean velocity. It has no simulated dark energy reservoir and cannot demonstrate dark depletion. The continuum dark flux is $\epsilon_0c^2\mathbf E_D\times\mathbf B_D+\epsilon_0\Omega_D^2\phi_D\mathbf A_D$. The slab example separately computes ordinary outgoing Poynting flux at a detector plane; there is no general dark-flux output API yet.

In a periodic closed run, the useful ledgers are more specific than a single total:

$$
\begin{aligned}
\dot U_D&=-\eta\int dx\,\mathbf J\cdot\mathbf E_D,\\
\frac{d}{dt}(U_{\rm particles}+U_{\rm EM})&=+\eta\int dx\,\mathbf J\cdot\mathbf E_D,\\
\dot U&=0.
\end{aligned}
$$

`DarkOutput.energy()` reports `dark_source_work`, `dark_work_residual` $=U_D(t)-U_D(0)-W_D(t)$, and `ordinary_work_residual` $=U_{\rm particles+EM}(t)-U_{\rm particles+EM}(0)+W_D(t)$. The two residuals add to the closed total-energy change. The initial energies and cumulative work survive sparse output and exact-parameter restart; `max_balance_error`, `max_ordinary_gauss`, and `max_dark_gauss` record every step, including unobserved steps. `store_particles=False` retains species kinetic and all field-energy histories without particle-position histories. Monitor both Gauss laws, $\mathbf B_D-\nabla\times\mathbf A_D$, and this work ledger: small total drift alone can hide cancellation between the sectors. The finite-step residuals need not vanish exactly for the explicit particle scheme.

Native archives include the dark mass, coupling, drive controls, grid and physical species parameters. Ordinary `load_state` rejects a changed experiment. `load_for_continuation` explicitly permits a dark-parameter jump from an old model to a new one, reprojects longitudinal dark E and starts a new energy/work ledger at the archived state; the clock and particle state remain continuous. An ordinary imposed electric field is rejected by the dark wrapper because its work is absent from this ledger.

For the homogeneous prescribed drive $E_D=D_0\cos\Omega t$ and an initially quiet cold plasma, the ordinary field obeys $\ddot E+\omega_p^2E=-\eta\omega_p^2D_0\cos\Omega t$. Its resonant limit is $E=-\eta\omega_pD_0t\sin(\omega_pt)/2$. The [drive example](../examples/dark_drive.py) evaluates the continuous, stable sinc form near resonance and checks the accumulated external-work balance. This is the same effective ordinary forcing; its linear-in-time amplitude is not exponential growth.

## Independent cold check

For a homogeneous cold plasma, fixed reference frequency $\omega_0$, $p=\omega_p^2/\omega_0^2$, and $\Omega_D=\omega_0$, normalize time by $\omega_0^{-1}$, E by a fixed $D_0$, A by $D_0/\omega_0$, and J by $\epsilon_0\omega_0D_0$. Then $y=(E,E_D,A_D,J)$ obeys

$$
\frac{dy}{d(\omega_0t)}=
\begin{pmatrix}0&0&0&-1\\0&0&1&-\eta\\0&-1&0&0\\p&\eta p&0&0\end{pmatrix}y.
$$

The [full example](../examples/dark_photon.py) compares PIC with an independently evaluated SciPy matrix exponential. This checks the coherent homogeneous response; it does not validate kinetic roots, nonlinear saturation, or outgoing photon conversion. At $\eta=0$ the ordinary run matches its JAX-in-Cell parent. At fixed nonzero $\eta$, $\Omega_D\to0$ leaves an interacting massless combination, so that is a different limit.
