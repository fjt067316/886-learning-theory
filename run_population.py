# infinitely many prompts (Section 3 and Figure 1)
import json
import numpy as np
import matplotlib.pyplot as plt
from model import *

d, N = 4, 10
Lam, Gam = make_problem(d, N, [0.5, 4 / 3, 13 / 6, 3.0], 0)
Gi = np.linalg.inv(Gam)
Li = np.linalg.inv(Lam)
G = np.linalg.norm(Gi)
lmin = min_loss(Lam, Gam)
rng = np.random.default_rng(1)
out = {}


def grads(U, u):
    return pop_grads(U, u, Lam, Gam)


def gap(U, u):
    return pop_loss(U, u, Lam, Gam) - lmin


# 1. check the loss formula against the full model with 400k prompts
out["mc_check"] = []
for k in range(4):
    U = 0.3 * rng.standard_normal((d, d))
    u = rng.uniform(0.3, 1.5)
    X, Y, A, y = sample_prompts(400000, d, N, Lam, rng)
    pred, _ = full_predict(X, Y, N, U, u, np.zeros(d), np.zeros(d))
    sq = (pred - y) ** 2
    out["mc_check"].append({"mc": 0.5 * sq.mean(),
                            "stderr": 0.5 * sq.std() / np.sqrt(len(sq)),
                            "formula": pop_loss(U, u, Lam, Gam) + 0.5 * np.trace(Lam)})

# 2. cross-term gradient is 0 when every task w comes with -w
X, Y, A, y = sample_prompts(2000, d, N, Lam, rng)
X = np.concatenate([X, X])
Y = np.concatenate([Y, -Y])
y = np.concatenate([y, -y])
U, u, h = rng.standard_normal((d, d)), 0.7, 1e-4


def full_loss(c_pv, c_kq):
    pred, _ = full_predict(X, Y, N, U, u, c_pv, c_kq)
    return 0.5 * np.mean((pred - y) ** 2)


z = np.zeros(d)
biggest = 0
for e in np.eye(d):
    g1 = (full_loss(z + h * e, z) - full_loss(z - h * e, z)) / (2 * h)
    g2 = (full_loss(z, z + h * e) - full_loss(z, z - h * e)) / (2 * h)
    biggest = max(biggest, abs(g1), abs(g2))
out["cross_grad_max"] = biggest

# 3. 150 random starts, check the end point and the rate of Theorems A and B
kappa = 2 / (np.trace(Gi @ Li) * np.trace(Li))
covered, drift, err, rate_ok = 0, 0, 0, True
for k in range(150):
    U0 = rng.standard_normal((d, d)) * 10 ** rng.uniform(-1.5, 0.8) / 4
    u0 = rng.normal() * 10 ** rng.uniform(-1.5, 0.8)
    delta = u0**2 - np.sum(U0**2)
    l0 = pop_loss(U0, u0, Lam, Gam)
    U, u, sol = flow(grads, U0, u0, T=4000)

    # lower bound on u^2 from Theorem A (delta > 0) or Theorem B (loss < 0)
    s = 0
    if delta > 0:
        s = delta
    if l0 < 0:
        c0 = -l0 / np.linalg.norm(Lam @ Lam)
        s = max(s, end_u2(delta, c0))
    if s > 0:
        covered += 1
        for t in np.linspace(0, 50, 200):
            z = sol.sol(t)
            if gap(z[:-1].reshape(d, d), z[-1]) > gap(U0, u0) * np.exp(-kappa * s * t) + 1e-9:
                rate_ok = False

    drift = max(drift, abs(u**2 - np.sum(U**2) - delta))
    err = max(err, np.linalg.norm(u * U - Gi), abs(u**2 - end_u2(delta, G)))
out["random_starts"] = {"n": 150, "covered": covered, "max_drift": drift,
                        "max_endpoint_err": err, "rate_bound_held": rate_ok}

# 4. Hessian trace at the end point, formula vs finite differences
alpha = np.trace(Lam) * np.trace(Gam @ Lam)
beta = np.trace(Lam @ Lam @ Gi)


def trace_formula(s):
    return s * alpha + beta / s


def trace_fd(U, u, h=1e-4):
    f0 = pop_loss(U, u, Lam, Gam)
    t = (pop_loss(U, u + h, Lam, Gam) - 2 * f0 + pop_loss(U, u - h, Lam, Gam)) / h**2
    for i in range(d):
        for j in range(d):
            E = np.zeros((d, d))
            E[i, j] = h
            t += (pop_loss(U + E, u, Lam, Gam) - 2 * f0 + pop_loss(U - E, u, Lam, Gam)) / h**2
    return t


hess = []
for delta in [-12, -8, -5, -3, -1, 0, 1, 3]:
    s = end_u2(delta, G)
    hess.append({"delta": delta, "formula": trace_formula(s), "fd": trace_fd(Gi / np.sqrt(s), np.sqrt(s))})
s_flat = np.sqrt(beta / alpha)
delta_flat = s_flat - G**2 / s_flat
out["hessian"] = {"points": hess, "delta_flat": delta_flat, "trace_flat": trace_formula(s_flat),
                  "trace_balanced": trace_formula(end_u2(0, G))}

# delta_flat < 0 for random spectra with d >= 2
worst = -np.inf
for k in range(2000):
    dd = int(rng.integers(2, 8))
    L2, G2 = make_problem(dd, int(rng.integers(1, 200)), 10 ** rng.uniform(-2, 2, dd), k)
    a = np.trace(L2) * np.trace(G2 @ L2)
    b = np.trace(L2 @ L2 @ np.linalg.inv(G2))
    g = np.linalg.norm(np.linalg.inv(G2))
    sf = np.sqrt(b / a)
    worst = max(worst, sf - g**2 / sf)
out["hessian"]["max_delta_flat_random_spectra"] = worst

# 5. figure
V0 = rng.standard_normal((d, d))
V0 /= np.linalg.norm(V0)
if np.trace(Lam @ Lam @ V0.T) < 0:
    V0 = -V0
fig, ax = plt.subplots(2, 2, figsize=(11, 9))

# training paths
x = np.linspace(0.05, 3.2, 300)
ax[0, 0].plot(x, G / x, "k", lw=2.5, label="minima: $u\\,\\|U\\|_F=\\|\\Gamma^{-1}\\|_F$")
starts = [(0.12, 0.12, "balanced (paper)"), (2.2, 0.6, "$\\delta>0$"),
          (3.0, 2.2, "$\\delta>0$, loss starts > 0"), (0.15, 2.0, "$\\delta<0$")]
for u0, s0, lab in starts:
    U, u, sol = flow(grads, V0 * s0, u0, T=400)
    zs = sol.sol(np.linspace(0, 400, 3000)).T
    ax[0, 0].plot(zs[:, -1], np.linalg.norm(zs[:, :-1], axis=1), lw=2, label=lab)
ax[0, 0].set(xlim=(0, 3.2), ylim=(0, 3.2), xlabel="$u$", ylabel="$\\|U\\|_F$", title="training paths ($d=4$)")
ax[0, 0].legend(fontsize=9, frameon=False)

# end point vs delta
sim_deltas = [-3.5, -2, -1, 0, 1, 2, 3.5]
sims = []
for delta in sim_deltas:
    s0 = np.sqrt(0.25 - delta) if delta < 0 else 0.5
    U, u, _ = flow(grads, V0 * s0, np.sqrt(s0**2 + delta), T=800)
    sims.append(u**2)
deltas = np.linspace(-4, 4, 200)
ax[0, 1].plot(deltas, end_u2(deltas, G), lw=2, label="formula")
ax[0, 1].plot(sim_deltas, sims, "o", label="simulation")
ax[0, 1].set(xlabel="imbalance $\\delta$", ylabel="final $u^2$", title="where training ends")
ax[0, 1].legend(frameon=False)

# speed vs delta, same U0 and only u0 changes
def stop(U, u):
    return gap(U, u) - 1e-8


out["speed"] = []
for delta in [-0.9999, -0.99, -0.5, 0, 1, 4]:
    U, u, sol = flow(grads, V0.copy(), np.sqrt(1 + delta), T=300, stop=stop)
    ts = np.linspace(0, sol.t[-1], 400)
    gaps = [gap(z[:-1].reshape(d, d), z[-1]) for z in sol.sol(ts).T]
    ax[1, 0].semilogy(ts, gaps, label=f"$\\delta={delta:g}$")
    out["speed"].append({"delta": delta, "u0": np.sqrt(1 + delta), "time": sol.t[-1]})
ax[1, 0].set(xlabel="training time", ylabel="loss gap", title="bigger $\\delta$ trains faster ($\\|U_0\\|_F=1$)")
ax[1, 0].legend(fontsize=9, frameon=False)

# Hessian trace vs delta
ds = np.linspace(-14, 4, 300)
ax[1, 1].plot(ds, trace_formula(end_u2(ds, G)), lw=2, label="formula")
ax[1, 1].plot([r["delta"] for r in hess], [r["fd"] for r in hess], "o", label="finite differences")
ax[1, 1].axvline(0, ls="--", c="g", lw=1)
ax[1, 1].axvline(delta_flat, ls="--", c="tab:orange", lw=1.5)
ax[1, 1].set(yscale="log", xlabel="imbalance $\\delta$", ylabel="Hessian trace at the end",
             title="same function, different sharpness")
ax[1, 1].legend(frameon=False)

fig.tight_layout()
fig.savefig("figures/fig_population.pdf")

json.dump(out, open("results/population.json", "w"), indent=1, default=float)
print(json.dumps(out, indent=1, default=float))
