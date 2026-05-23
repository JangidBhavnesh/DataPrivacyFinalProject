
import os
import math
import numpy as np
import matplotlib.pyplot as plt


def rr_eta(theta, eps):
    """
    Probability P(Z=1) after randomized response when X ~ Bernoulli(theta).

    Randomized response:
        P(Z=X)   = exp(eps)/(1+exp(eps))
        P(Z=1-X) = 1/(1+exp(eps))

    This gives:
        eta(theta, eps) = q + (p-q)*theta
                        = q + tanh(eps/2)*theta
    """
    p_minus_q = np.tanh(eps / 2.0)
    q = 1.0 / (1.0 + np.exp(eps))
    return q + p_minus_q * theta


def lambda_scaling(n, a, eps, d):
    """
    Scaling variable from the report:
        lambda = n a^2 tanh^2(eps/2) / log(d)
    """
    return n * (a ** 2) * (np.tanh(eps / 2.0) ** 2) / np.log(d)


def log_binomial_likelihood_ratio_counts(T, n, eta1, eta0):
    """
    For each coordinate j, compute

        log r_j = log P(T_j | j active) - log P(T_j | j inactive)

    where
        T_j | active   ~ Binomial(n, eta1)
        T_j | inactive ~ Binomial(n, eta0)

    The binomial coefficient cancels in the likelihood ratio.
    """
    tiny = 1e-12
    eta1 = np.clip(eta1, tiny, 1.0 - tiny)
    eta0 = np.clip(eta0, tiny, 1.0 - tiny)

    return (
        T * np.log(eta1 / eta0)
        + (n - T) * np.log((1.0 - eta1) / (1.0 - eta0))
    )


def posterior_entropy_fixed_s(logw, s):
    """
    Exact posterior entropy over all supports of size s.

    Posterior:
        P(S | Z) proportional to product_{j in S} w_j

    where logw_j is the coordinate log-likelihood ratio.

    This avoids direct enumeration over choose(d,s) supports. It uses an
    elementary-symmetric-polynomial dynamic program to compute the exact
    partition function and coordinate inclusion probabilities in O(d*s^2).

    Entropy:
        H = log Z - E_posterior[log weight(S)]
          = log Z - sum_j P(j in S | Z) logw_j
    """
    logw = np.asarray(logw, dtype=float)
    d = len(logw)
    Hmax = math.log(math.comb(d, s))

    # Shift weights to avoid overflow. This does not change the entropy because
    # all supports have exactly s coordinates.
    shift = np.max(logw)
    w = np.exp(logw - shift)

    # Forward DP:
    # F[j, k] = total shifted weight for choosing k items among coordinates 0,...,j-1
    F = np.zeros((d + 1, s + 1), dtype=float)
    F[0, 0] = 1.0
    for j in range(1, d + 1):
        F[j, 0] = 1.0
        upper = min(j, s)
        F[j, 1:upper + 1] = (
            F[j - 1, 1:upper + 1]
            + w[j - 1] * F[j - 1, 0:upper]
        )

    # Backward DP:
    # B[j, k] = total shifted weight for choosing k items among coordinates j,...,d-1
    B = np.zeros((d + 1, s + 1), dtype=float)
    B[d, 0] = 1.0
    for j in range(d - 1, -1, -1):
        B[j, 0] = 1.0
        upper = min(d - j, s)
        B[j, 1:upper + 1] = (
            B[j + 1, 1:upper + 1]
            + w[j] * B[j + 1, 0:upper]
        )

    Z_shifted = F[d, s]

    # Rare numerical fallback if shifted weights underflow too strongly.
    if not np.isfinite(Z_shifted) or Z_shifted <= 0:
        return posterior_entropy_fixed_s_logspace(logw, s)

    logZ = math.log(Z_shifted) + s * shift

    # Coordinate inclusion probabilities:
    # pi_j = w_j * sum_k F[j,k] B[j+1,s-1-k] / Z
    pi = np.zeros(d)
    for j in range(d):
        acc = 0.0
        for k_left in range(s):
            k_right = s - 1 - k_left
            if k_left <= j and k_right <= d - j - 1:
                acc += F[j, k_left] * B[j + 1, k_right]
        pi[j] = w[j] * acc / Z_shifted

    expected_log_weight = np.dot(pi, logw)
    H = logZ - expected_log_weight

    # Numerical guard
    H = min(max(H, 0.0), Hmax)
    return H


def posterior_entropy_fixed_s_logspace(logw, s):
    """
    Slower log-space fallback for extreme numerical cases.
    """
    logw = np.asarray(logw, dtype=float)
    d = len(logw)
    neg_inf = -np.inf

    F = np.full((d + 1, s + 1), neg_inf)
    F[0, 0] = 0.0
    for j in range(1, d + 1):
        F[j, 0] = 0.0
        upper = min(j, s)
        for k in range(1, upper + 1):
            F[j, k] = np.logaddexp(F[j - 1, k], F[j - 1, k - 1] + logw[j - 1])

    B = np.full((d + 1, s + 1), neg_inf)
    B[d, 0] = 0.0
    for j in range(d - 1, -1, -1):
        B[j, 0] = 0.0
        upper = min(d - j, s)
        for k in range(1, upper + 1):
            B[j, k] = np.logaddexp(B[j + 1, k], B[j + 1, k - 1] + logw[j])

    logZ = F[d, s]

    pi = np.zeros(d)
    for j in range(d):
        log_num = -np.inf
        for k_left in range(s):
            k_right = s - 1 - k_left
            if k_left <= j and k_right <= d - j - 1:
                log_num = np.logaddexp(log_num, F[j, k_left] + logw[j] + B[j + 1, k_right])
        pi[j] = np.exp(log_num - logZ)

    H = logZ - np.dot(pi, logw)
    Hmax = math.log(math.comb(d, s))
    return min(max(H, 0.0), Hmax)


def run_simulation(
    d=100,
    s=5,
    theta0=0.10,
    a=0.08,
    eps_values=None,
    n_values=(500, 1000, 2000),
    reps=150,
    seed=7,
):
    """
    Main simulation.

    To keep the simulation fast, we sample the sufficient statistics directly:
        T_j = sum_i Z_ij

    rather than explicitly storing the full n x d matrix Z.

    This is equivalent for the top-count adversary and posterior calculation,
    because T_j is binomial under active/null coordinates.
    """
    if eps_values is None:
        eps_values = np.linspace(0.1, 6.0, 30)

    rng = np.random.default_rng(seed)
    rows = []

    for n in n_values:
        for eps in eps_values:
            eta0 = rr_eta(theta0, eps)
            eta1 = rr_eta(theta0 + a, eps)
            lam = lambda_scaling(n, a, eps, d)
            Hmax = math.log(math.comb(d, s))

            mrec_rep = np.empty(reps)
            mpriv_rep = np.empty(reps)

            for r in range(reps):
                # By symmetry, take the true support to be coordinates 0,...,s-1.
                # Randomizing the support gives the same distribution but is slower.
                T_active = rng.binomial(n, eta1, size=s)
                T_null = rng.binomial(n, eta0, size=d - s)
                T = np.concatenate([T_active, T_null])

                # Recovery overlap: top-s count estimator
                top_s = np.argpartition(T, -s)[-s:]
                overlap = np.sum(top_s < s)
                mrec_rep[r] = overlap / s

                # Entropy-collapse order parameter
                logw = log_binomial_likelihood_ratio_counts(T, n, eta1, eta0)
                H = posterior_entropy_fixed_s(logw, s)
                mpriv_rep[r] = 1.0 - H / Hmax

            rows.append(
                {
                    "n": n,
                    "d": d,
                    "s": s,
                    "theta0": theta0,
                    "a": a,
                    "eps": eps,
                    "lambda": lam,
                    "mrec_mean": float(np.mean(mrec_rep)),
                    "mrec_se": float(np.std(mrec_rep, ddof=1) / np.sqrt(reps)),
                    "mpriv_mean": float(np.mean(mpriv_rep)),
                    "mpriv_se": float(np.std(mpriv_rep, ddof=1) / np.sqrt(reps)),
                }
            )

    return rows


def save_results_csv(rows, filename):
    import csv

    fieldnames = list(rows[0].keys())
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows, outdir="figures"):
    os.makedirs(outdir, exist_ok=True)

    n_values = sorted(set(row["n"] for row in rows))

    # Figure 1: recovery overlap vs epsilon
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for n in n_values:
        sub = sorted([row for row in rows if row["n"] == n], key=lambda x: x["eps"])
        x = np.array([row["eps"] for row in sub])
        y = np.array([row["mrec_mean"] for row in sub])
        yerr = np.array([row["mrec_se"] for row in sub])
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=2, label=f"n={n}")
    ax.set_xlabel(r"Privacy budget $\epsilon$")
    ax.set_ylabel(r"Recovery overlap $m_{\rm rec}$")
    ax.set_title("Figure 1: Recovery overlap vs privacy budget")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig1_recovery_vs_epsilon.png"), dpi=300)
    plt.close(fig)

    # Figure 2: recovery overlap vs lambda
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for n in n_values:
        sub = sorted([row for row in rows if row["n"] == n], key=lambda x: x["lambda"])
        x = np.array([row["lambda"] for row in sub])
        y = np.array([row["mrec_mean"] for row in sub])
        yerr = np.array([row["mrec_se"] for row in sub])
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=2, label=f"n={n}")
    ax.set_xscale("log")
    ax.set_xlabel(r"Scaling variable $\lambda = n a^2 \tanh^2(\epsilon/2)/\log d$")
    ax.set_ylabel(r"Recovery overlap $m_{\rm rec}$")
    ax.set_title("Figure 2: Recovery overlap collapses against scaling variable")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig2_recovery_vs_lambda.png"), dpi=300)
    plt.close(fig)

    # Figure 3: entropy collapse vs lambda
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    for n in n_values:
        sub = sorted([row for row in rows if row["n"] == n], key=lambda x: x["lambda"])
        x = np.array([row["lambda"] for row in sub])
        y = np.array([row["mpriv_mean"] for row in sub])
        yerr = np.array([row["mpriv_se"] for row in sub])
        ax.errorbar(x, y, yerr=yerr, marker="o", capsize=2, label=f"n={n}")
    ax.set_xscale("log")
    ax.set_xlabel(r"Scaling variable $\lambda$")
    ax.set_ylabel(r"Entropy-collapse exposure $m_{\rm priv}$")
    ax.set_title("Figure 3: Posterior entropy collapse vs scaling variable")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_entropy_collapse_vs_lambda.png"), dpi=300)
    plt.close(fig)

    # Figure 4: susceptibility-like finite-difference slope
    # Use the middle n value as the default finite-size slice.
    n_mid = n_values[len(n_values) // 2]
    sub = sorted([row for row in rows if row["n"] == n_mid], key=lambda x: x["lambda"])
    x = np.array([row["lambda"] for row in sub])
    mrec = np.array([row["mrec_mean"] for row in sub])
    mpriv = np.array([row["mpriv_mean"] for row in sub])

    # A derivative with respect to log(lambda) is more visually useful here because
    # lambda spans orders of magnitude. This highlights the finite-size crossover
    # region instead of over-weighting the tiny-lambda regime.
    logx = np.log(x)
    chi_rec = np.gradient(mrec, logx)
    chi_priv = np.gradient(mpriv, logx)

    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    ax.plot(x, chi_rec, marker="o", label=r"$d m_{\rm rec}/d\log\lambda$")
    ax.plot(x, chi_priv, marker="s", label=r"$d m_{\rm priv}/d\log\lambda$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Scaling variable $\lambda$")
    ax.set_ylabel(r"Finite-size susceptibility $dm/d\log\lambda$")
    ax.set_title(f"Figure 4: Critical region from susceptibility-like peak, n={n_mid}")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_susceptibility_vs_lambda.png"), dpi=300)
    plt.close(fig)


def main():
    outdir = "ldp_simulation_outputs"
    os.makedirs(outdir, exist_ok=True)

    rows = run_simulation(
        d=100,
        s=5,
        theta0=0.10,
        a=0.08,
        eps_values=np.linspace(0.1, 6.0, 30),
        n_values=(500, 1000, 2000),
        reps=150,
        seed=7,
    )

    save_results_csv(rows, os.path.join(outdir, "simulation_results.csv"))
    make_figures(rows, outdir=outdir)

    print(f"Done. Results and figures saved in: {outdir}")


if __name__ == "__main__":
    main()
