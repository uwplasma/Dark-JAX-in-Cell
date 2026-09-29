# A density optimum with a known answer

The calibration keeps the physical reference frequency $\omega_0=10^9$ rad/s, box, time step, initial dark energy and coupling fixed. Its control is $p=n/n_{\rm ref}=\omega_p^2/\omega_0^2\in[0.5,1.5]$. The dark field starts with a small homogeneous transverse electric amplitude $D_0=10^{-5}$ V/m, and $\eta=0.05$. The physical time window is **fixed** at $45\le\omega_0t\le75$ for every density.

The objective is the mean coherent ordinary electric plus cold bulk energy, normalized by the fixed initial dark energy:

$$
\mathcal C(p)=\frac1{30}\int_{45}^{75}
\left[\left(\frac{E_y}{D_0}\right)^2+
\frac{1}{p}\left(\frac{J_y}{\epsilon_0\omega_0D_0}\right)^2\right]d(\omega_0t).
$$

The $1/p$ metric has a derivative of its own; dropping it changes the optimum. The JAX objective scans the **production** Maxwell–Proca PIC transition and retains only the state and a scalar accumulator. Particle weights and the initial loading depend on $p$ inside the differentiated function. SciPy's matrix exponential, Fréchet derivative and Gauss–Legendre quadrature supply an independent continuous cold reference.

The full PIC optimum is **{{ calibration_pic_p }}** with objective **{{ calibration_pic_objective }}**. Halving the step gives **{{ calibration_refined_p }}** and **{{ calibration_refined_objective }}**. The independent cold optimum is **{{ calibration_cold_p }}** with objective **{{ calibration_cold_objective }}**. At $p=0.97$, the PIC-versus-Fréchet gradient error falls from **{{ calibration_gradient_error }}** to **{{ calibration_refined_gradient_error }}** under this refinement; the objective and optimum errors also approach the reference. The minimum absolute discrepancy of a centred finite-difference sweep against PIC AD is **{{ calibration_fd_error }}**. The window integrator uses trapezoids with partial boundary steps so every density trial sees the same physical interval. The figure compares the complete scan and the gradient sweep.

![Density objective and gradient checks](_static/figures/density_calibration/figure.png)

Three bounded gradient starts are retained in the [run record](_static/figures/density_calibration/run.json). The start at $p=1.25$ reaches a **lower local maximum**; the best of multiple starts and the independent uniform scan identify the global candidate. This objective measures coherent plasma exchange, not outgoing photons or irreversible heating. The independent physical optimum is not used as a hardcoded PIC target. The quick preset uses a shorter window and must not be cited as this validation.
