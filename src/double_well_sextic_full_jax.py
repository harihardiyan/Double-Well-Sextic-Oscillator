
# double_well_sextic_curvature_alive_centered_pr_jax_full.py
# Full JAX, x64: symmetric double-well with negative quartic and positive sextic
# H = p^2/(2m) + 1/2 m ω^2 x^2 + α x^4 + γ x^6  (α<0, γ>0)
# ω-adaptive basis, centered moments (skew/kurt/excess), PR and entropy, side sampling around boundary

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

hbar = 1.0
m    = 1.0

# =========================
# Operators and Hamiltonian (ω-adaptive)
# =========================
def create_fock_ops(N):
    n = jnp.arange(N)
    s = jnp.sqrt(n[1:])
    a    = jnp.zeros((N, N), dtype=jnp.complex128)
    adag = jnp.zeros((N, N), dtype=jnp.complex128)
    a    = a.at[1:, :-1].set(jnp.diag(s).astype(jnp.complex128))
    adag = adag.at[:-1, 1:].set(jnp.diag(s).astype(jnp.complex128))
    return a, adag

def x_op(N, omega_ref):
    a, adag = create_fock_ops(N)
    scale = jnp.sqrt(hbar/(2*m*omega_ref))
    return scale * (a + adag)

def p_op(N, omega_ref):
    a, adag = create_fock_ops(N)
    scale = 1j*jnp.sqrt(m*hbar*omega_ref/2.0)
    return scale * (adag - a)

def H_sextic_double_well(N, omega, alpha, gamma):
    # Build operators with omega_ref = omega (adaptive basis)
    X = x_op(N, omega_ref=omega)
    P = p_op(N, omega_ref=omega)
    X2 = X @ X
    X3 = X2 @ X
    X4 = X2 @ X2
    X6 = X4 @ X2
    H = (P @ P)/(2*m) + 0.5*m*(omega**2)*X2 + alpha*X4 + gamma*X6
    return H, X, X2, X3, X4, X6

def ground_state(N, omega, alpha, gamma):
    H, X, X2, X3, X4, X6 = H_sextic_double_well(N, omega, alpha, gamma)
    E, U = jnp.linalg.eigh(H)
    psi0 = U[:, 0]
    psi0 = psi0 / jnp.linalg.norm(psi0)
    return X, X2, X3, X4, X6, psi0, E

# =========================
# Centered moments, PR, entropy, curvature
# =========================
def centered_moments(N, omega, alpha, gamma):
    X, X2, X3, X4, X6, psi0, _ = ground_state(N, omega, alpha, gamma)
    # raw moments
    x1 = jnp.real(jnp.vdot(psi0, X  @ psi0))
    x2 = jnp.real(jnp.vdot(psi0, X2 @ psi0))
    x3 = jnp.real(jnp.vdot(psi0, X3 @ psi0))
    x4 = jnp.real(jnp.vdot(psi0, X4 @ psi0))
    # centered moments using μ = x1
    mu = x1
    var = jnp.maximum(x2 - mu**2, 1e-18)
    m3c = x3 - 3*mu*x2 + 2*mu**3
    m4c = x4 - 4*mu*x3 + 6*mu**2*x2 - 3*mu**4
    skew = m3c / (var**1.5)
    kurt = m4c / (var**2)           # Gaussian = 3
    excess = kurt - 3.0             # Gaussian = 0
    return mu, var, skew, kurt, excess, x2, x4

def delocalization_metrics(psi):
    p = jnp.abs(psi)**2
    p = p / jnp.sum(p)
    PR = 1.0 / jnp.sum(p**2)
    entropy = -jnp.sum(jnp.where(p>0, p*jnp.log(p), 0.0))
    return PR, entropy

def curvature_eigen_min(N, omega, alpha, gamma):
    X, X2, X3, X4, X6, psi0, _ = ground_state(N, omega, alpha, gamma)
    x2 = jnp.real(jnp.vdot(psi0, X2 @ psi0))
    x4 = jnp.real(jnp.vdot(psi0, X4 @ psi0))
    K_xx = m*(omega**2) + 12.0*alpha*x2 + 30.0*gamma*x4
    K_pp = 1.0/m
    K = jnp.array([[K_xx, 0.0],[0.0, K_pp]], dtype=jnp.float64)
    return jnp.min(jnp.linalg.eigvalsh(K))

# =========================
# Validated boundary (no-NaN collection)
# =========================
def find_boundary_alpha_valid(N, omega, gamma, alpha_lo, alpha_hi, n_grid=128):
    alphas = jnp.linspace(alpha_lo, alpha_hi, n_grid)

    def f(a):
        return curvature_eigen_min(N, omega, a, gamma)

    emin = jax.vmap(f)(alphas)
    signs = jnp.sign(emin)
    cross = jnp.where(signs[:-1] * signs[1:] < 0, 1, 0)
    if int(jnp.sum(cross)) == 0:
        return None  # no boundary in bracket

    idx = int(jnp.argmax(cross))
    alo = float(alphas[idx]); ahi = float(alphas[idx+1])

    for _ in range(60):
        amid = 0.5*(alo + ahi)
        e_mid = float(curvature_eigen_min(N, omega, amid, gamma))
        if e_mid >= 0.0:
            ahi = amid
        else:
            alo = amid
        if (ahi - alo) < 1e-4:
            break
    return 0.5*(alo + ahi)

# =========================
# Observables + side sampling
# =========================
def observables_full(N, omega, alpha, gamma):
    X, X2, X3, X4, X6, psi0, E = ground_state(N, omega, alpha, gamma)
    mu, var, skew, kurt, excess, x2r, x4r = centered_moments(N, omega, alpha, gamma)
    PR, entropy = delocalization_metrics(psi0)
    return {
        "mu": float(mu), "var": float(var), "skew": float(skew),
        "kurt": float(kurt), "excess": float(excess),
        "x2": float(x2r), "x4": float(x4r),
        "PR": float(PR), "entropy": float(entropy),
        "E0": float(E[0])
    }

# =========================
# Main: boundary + side sampling (lebih hidup)
# =========================
if __name__ == "__main__":
    # Konfigurasi yang memperlihatkan dinamika bentuk:
    N = 96
    gamma = 0.0016         # sextic lebih lemah agar quartic -α mempengaruhi bentuk
    omega_grid = jnp.linspace(0.6, 1.6, 31)
    alpha_bracket = (-6.0, 1.2)
    delta_alpha = 0.05     # sampling ±Δα cukup jauh untuk melihat perubahan bentuk

    omegas_valid = []
    alpha_star   = []

    # summary dict dengan key konsisten
    keys = ["mu","var","skew","kurt","excess","PR","entropy"]
    summary = {f"{k}_b":[] for k in keys}
    summary.update({f"{k}_m":[] for k in keys})
    summary.update({f"{k}_p":[] for k in keys})

    for w in omega_grid:
        a_star = find_boundary_alpha_valid(N, float(w), gamma, alpha_bracket[0], alpha_bracket[1], n_grid=128)
        if a_star is None:
            continue
        omegas_valid.append(float(w))
        alpha_star.append(float(a_star))

        # Boundary
        ob_b = observables_full(N, float(w), float(a_star), gamma)
        # Side -Δα
        ob_m = observables_full(N, float(w), float(a_star - delta_alpha), gamma)
        # Side +Δα
        ob_p = observables_full(N, float(w), float(a_star + delta_alpha), gamma)

        for k in keys:
            summary[f"{k}_b"].append(ob_b[k])
            summary[f"{k}_m"].append(ob_m[k])
            summary[f"{k}_p"].append(ob_p[k])

    omegas_valid = jnp.array(omegas_valid)
    alpha_star   = jnp.array(alpha_star)

    def arr(key): return jnp.array(summary[key])

    print(f"[Boundary count] found={omegas_valid.shape[0]} of {omega_grid.shape[0]} omegas")
    if omegas_valid.shape[0] > 0:
        print(f"[Boundary] ω range={float(jnp.min(omegas_valid)):.3f}..{float(jnp.max(omegas_valid)):.3f} | "
              f"α* range={float(jnp.min(alpha_star)):.3f}..{float(jnp.max(alpha_star)):.3f}")

        # Ringkasan variasi bentuk di boundary dan sisi (lebih hidup)
        print("[Centered observables @ boundary]")
        print(f"μ: mean(|μ_b|)={float(jnp.mean(jnp.abs(arr('mu_b')))):.4e} | "
              f"Var: min={float(jnp.min(arr('var_b'))):.4e}, max={float(jnp.max(arr('var_b'))):.4e} | "
              f"Kurt: min={float(jnp.min(arr('kurt_b'))):.4f}, max={float(jnp.max(arr('kurt_b'))):.4f} | "
              f"Excess: min={float(jnp.min(arr('excess_b'))):.4f}, max={float(jnp.max(arr('excess_b'))):.4f} | "
              f"PR: min={float(jnp.min(arr('PR_b'))):.4f}, max={float(jnp.max(arr('PR_b'))):.4f} | "
              f"Entropy: min={float(jnp.min(arr('entropy_b'))):.4f}, max={float(jnp.max(arr('entropy_b'))):.4f}")

        print("[Centered observables @ α* - Δα]")
        print(f"μ: mean(|μ_m|)={float(jnp.mean(jnp.abs(arr('mu_m')))):.4e} | "
              f"Var: min={float(jnp.min(arr('var_m'))):.4e}, max={float(jnp.max(arr('var_m'))):.4e} | "
              f"Kurt: min={float(jnp.min(arr('kurt_m'))):.4f}, max={float(jnp.max(arr('kurt_m'))):.4f} | "
              f"Excess: min={float(jnp.min(arr('excess_m'))):.4f}, max={float(jnp.max(arr('excess_m'))):.4f} | "
              f"PR: min={float(jnp.min(arr('PR_m'))):.4f}, max={float(jnp.max(arr('PR_m'))):.4f} | "
              f"Entropy: min={float(jnp.min(arr('entropy_m'))):.4f}, max={float(jnp.max(arr('entropy_m'))):.4f}")

        print("[Centered observables @ α* + Δα]")
        print(f"μ: mean(|μ_p|)={float(jnp.mean(jnp.abs(arr('mu_p')))):.4e} | "
              f"Var: min={float(jnp.min(arr('var_p'))):.4e}, max={float(jnp.max(arr('var_p'))):.4e} | "
              f"Kurt: min={float(jnp.min(arr('kurt_p'))):.4f}, max={float(jnp.max(arr('kurt_p'))):.4f} | "
              f"Excess: min={float(jnp.min(arr('excess_p'))):.4f}, max={float(jnp.max(arr('excess_p'))):.4f} | "
              f"PR: min={float(jnp.min(arr('PR_p'))):.4f}, max={float(jnp.max(arr('PR_p'))):.4f} | "
              f"Entropy: min={float(jnp.min(arr('entropy_p'))):.4f}, max={float(jnp.max(arr('entropy_p'))):.4f}")
    else:
        print("No boundary found in the given bracket. Consider widening α_bracket or lowering γ.")
