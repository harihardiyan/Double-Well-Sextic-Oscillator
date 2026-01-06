

# Delocalization and Stability in Symmetric Double-Well Sextic Oscillators

![Python Version](https://img.shields.io/badge/python-3.9%2B-blue?logo=python&logoColor=white)
![JAX](https://img.shields.io/badge/Accelerated_by-JAX-orange?logo=google&logoColor=white)
![Physics](https://img.shields.io/badge/Field-Quantum_Statistical_Physics-red)
![License](https://img.shields.io/badge/license-MIT-green)

## 1. Abstract
This repository explores the quantum phase transition and delocalization metrics of a **Symmetric Double-Well Sextic Oscillator**. By employing an $\omega$-adaptive Fock basis, the framework accurately captures the ground-state properties in regimes where the quartic term ($\alpha < 0$) creates a dual-minimum potential. The study utilizes advanced statistical measures, including **Participation Ratio (PR)**, **Shannon Entropy**, and **Centered Moments**, to quantify the spread of the wave function and its structural stability via curvature matrix analysis.

## 2. Project Structure (Root Tree)
```text
Double-Well-Sextic-Oscillator/
├── .gitignore               # Excludes JAX/XLA cache and binaries
├── LICENSE                  # MIT License (2026)
├── README.md                # Professional scientific documentation
├── requirements.txt         # JAX and NumPy dependencies
└── src/                     
    └── double_well_sextic_full_jax.py  # Core adaptive simulation engine
```

---

<p align="center">
  <b>Author:</b> Hari Hardiyan <br>
  <b>Email:</b> <a href="mailto:lorozloraz@gmail.com">lorozloraz@gmail.com</a> <br><br>
  <b>Lead AI Development:</b> AI Tamer <br>
  <b>Assistant:</b> Microsoft Copilot
</p>

---

## 3. Physical Model: The Double-Well Potential
The Hamiltonian represents a symmetric double-well system when the quartic non-linearity is negative:
$$\hat{H} = \frac{\hat{p}^2}{2m} + \frac{1}{2}m\omega^2\hat{x}^2 + \alpha\hat{x}^4 + \gamma\hat{x}^6$$
where:
*   **$\alpha < 0$**: Generates the potential barrier at the origin.
*   **$\gamma > 0$**: Ensures asymptotic stability.
*   **$\omega$-adaptive Basis**: The basis functions are dynamically scaled to the reference frequency $\omega$ to minimize truncation errors in the double-well regime.

## 4. Methodology & Advanced Observables
### 4.1 Delocalization Metrics
To measure the "quantumness" and the spread of the state across the Fock basis, we implement:
*   **Participation Ratio (PR)**: $PR = 1 / \sum |c_n|^4$, indicating the number of basis states significantly contributing to the ground state.
*   **Information Entropy ($S$):** Calculated from the probability distribution in the energy eigenbasis.

### 4.2 Centered Statistical Moments
Unlike raw moments, this engine calculates **Centered Moments** ($\mu_n$) to isolate the shape of the wave function:
*   **Variance ($\sigma^2$):** Measures the spatial spread.
*   **Excess Kurtosis ($\gamma_2$):** Quantifies the "flatness" or "peakedness" relative to a Gaussian distribution.

### 4.3 Side Sampling Analysis
The engine performs **Side Sampling** at $\alpha^* \pm \Delta\alpha$ around the stability boundary. This allows for a microscopic view of how the wave function's geometry (PR, Entropy, and Kurtosis) evolves as the system approaches structural collapse.

## 5. Numerical Performance
*   **High-Precision Bisection:** 60-step iterative refinement for the critical $\alpha^*$.
*   **X64 JAX Backend:** Ensures numerical stability for high-order moments ($x^4$, $x^6$).
*   **Adaptive Scaling:** Dynamically adjusts the length scale of the position operator based on $\omega$.

## 6. Representative Results
Execution with $\gamma=0.0016$ across an $\omega$ range $[0.6, 1.6]$ yields:

```text
[Boundary] ω range=0.600..1.600 | α* range=-1.147..-0.432
[Centered observables @ boundary]
μ: ~0 | Var: 107.5..286.7 | Kurt: 1.0000 | Excess: -2.0000
PR: 5.339..5.402 | Entropy: 1.821..1.832
```
*Note: An Excess Kurtosis of -2.0 indicates a highly platykurtic distribution, characteristic of the bi-modal structure in double-well potentials.*

## 7. License
Licensed under the **MIT License**. (See [LICENSE](LICENSE) for details).

## 8. Citation
> **Hardiyan, H. (2026).** *Delocalization and Stability in Double-Well Sextic Oscillators via JAX.* GitHub: [harihardiyan/Double-Well-Sextic-Oscillator](https://github.com/harihardiyan/Double-Well-Sextic-Oscillator).

---
*Developed under the AI Tamer initiative with assistance from Microsoft Copilot.*

---
