# finitely many training prompts (Sections 4 and 5, Figures 2 to 4)
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from model import *

d, N = 4, 10
Lam, Gam = make_problem(d, N, [0.5, 4 / 3, 13 / 6, 3.0], 0)
best_test = min_loss(Lam, Gam) + 0.5 * np.trace(Lam)  # test loss of P = Gamma^-1
rng = np.random.default_rng(2)
out = {}


def rand_U(size, rng):
    U = rng.standard_normal((d, d))
    return U * size / np.linalg.norm(U)


def train(U0, u0, A, y, T, stop_at=None):
    def grads(U, u):
        return fit_grads(U, u, A, y)

    def stop(U, u):
        return fit_loss(U, u, A, y) - stop_at

    return flow(grads, U0, u0, T, stop if stop_at else None)


def trace_at_fit(U, u, A):
    # Hessian trace of the training loss, sum over U entries plus the u entry
    s = np.sum(A * U, axis=(1, 2))
    return np.mean(u**2 * np.sum(A**2, axis=(1, 2)) + s**2)


# 1. Theorem E formula vs simulated training, 40 random starts
errs, flips = [], 0
for B in [3, 6, 10, 14]:
    for k in range(10):
        X, Y, A, y = sample_prompts(B, d, N, Lam, rng)
        U0 = rand_U(10 ** rng.uniform(-1, 0.5), rng)
        u0 = rng.choice([-1, 1]) * 10 ** rng.uniform(-1.3, 0.5)
        U, u, _ = train(U0, u0, A, y, T=1e5)
        errs.append(np.linalg.norm(u * U - learned_predictor(U0, u0, A, y, sign=u)))
        if np.sign(u) != np.sign(u0):
            flips += 1
out["formula_check"] = {"runs": len(errs), "max_err": max(errs), "sign_flips": flips}

# 2. test error vs B for balanced starts of different scales
Bs = [2, 4, 6, 8, 10, 12, 14, 15, 16, 18, 20, 24, 32, 48]
scales = [0.01, 0.1, 0.3, 1.0, 2.0]
curves = {s: [] for s in scales}
for B in Bs:
    losses = {s: [] for s in scales}
    for t in range(40):
        X, Y, A, y = sample_prompts(B, d, N, Lam, rng)
        V = rand_U(1.0, rng)  # same prompts and direction for every scale
        for s in scales:
            P = learned_predictor(V * s, s, A, y, sign=1)
            losses[s].append(excess_test_loss(P, Lam, Gam))
    for s in scales:
        curves[s].append(np.percentile(losses[s], [25, 50, 75]))
out["test_vs_B"] = {str(s): [list(c) for c in curves[s]] for s in scales}

zero = -min_loss(Lam, Gam)  # excess loss of always predicting 0
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for s in scales:
    c = np.array(curves[s])
    ax.plot(Bs, c[:, 1], "o-", ms=4, label=f"init scale {s:g}")
    ax.fill_between(Bs, c[:, 0], c[:, 2], alpha=0.12)
ax.axhline(zero, ls=":", c="gray")
ax.text(48, zero * 1.15, "always predict 0", ha="right", color="gray")
ax.axvline(d * d, ls="--", c="gray")
ax.set_yscale("log")
ax.set(xlabel="number of training prompts B", ylabel="excess in-context test loss")
ax.legend(fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig("figures/fig_test_vs_B.pdf")

# 3. speed vs generalization, B = 8, only u0 changes
u0s = [0.01, 0.03, 0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]
runs = []
for t in range(20):
    X, Y, A, y = sample_prompts(8, d, N, Lam, rng)
    base = excess_test_loss(min_norm_fit(A, y), Lam, Gam)
    V = rand_U(1.0, rng)
    for size in [0.3, 1.0]:
        for u0 in u0s:
            U, u, sol = train(V * size, u0, A, y, T=20000, stop_at=1e-8)
            P = learned_predictor(V * size, u0, A, y, sign=u)
            runs.append({"trial": t, "size": size, "u0": u0, "time": sol.t[-1],
                         "rel_test": excess_test_loss(P, Lam, Gam) / base,
                         "trace": trace_at_fit(U, u, A)})
out["speed_vs_generalization"] = runs

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for size in [0.3, 1.0]:
    times = [np.median([r["time"] for r in runs if r["size"] == size and r["u0"] == u0]) for u0 in u0s]
    tests = [np.median([r["rel_test"] for r in runs if r["size"] == size and r["u0"] == u0]) for u0 in u0s]
    ax[0].loglog(u0s, times, "o-", label=f"$\\|U_0\\|_F={size:g}$")
    ax[1].semilogx(u0s, tests, "o-", label=f"$\\|U_0\\|_F={size:g}$")
ax[0].set(xlabel="initial value weight $u_0$", ylabel="training time to fit", title="larger $u_0$ trains faster")
ax[1].set(xlabel="initial value weight $u_0$", ylabel="test loss / min-norm test loss",
          title="but keeps more of the initialization")
ax[1].axhline(1, ls=":", c="gray")
ax[0].legend(frameon=False)
ax[1].legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/fig_speed_vs_generalization.pdf")

# 4. is the flattest fit the best one?
flat_best, flat_rel, best_rel = 0, [], []
for t in range(20):
    rs = [r for r in runs if r["trial"] == t]
    flat = min(rs, key=lambda r: r["trace"])
    best = min(rs, key=lambda r: r["rel_test"])
    if flat is best:
        flat_best += 1
    flat_rel.append(flat["rel_test"])
    best_rel.append(best["rel_test"])
out["sharpness"] = {"flattest_is_best": flat_best, "flat_rel_median": np.median(flat_rel),
                    "best_rel_median": np.median(best_rel)}

# trace vs test loss, one line per delta, the start grows along each line
rng2 = np.random.default_rng(5)
draws = []
for t in range(20):
    w = rng2.standard_normal((d, d))
    X, Y, A, y = sample_prompts(8, d, N, Lam, rng2)
    draws.append((A, y, w / np.linalg.norm(w)))

sizes = [0.1, 0.2, 0.4, 0.7, 1.0, 1.5, 2.0, 3.0]
deltas = [-0.5, 0.0, 1.0, 4.0]
sharp = {}
for delta in deltas:
    sharp[delta] = []
    for size in sizes:
        if size**2 + delta <= 0:
            continue
        u0 = np.sqrt(size**2 + delta)
        tr, te = [], []
        for A, y, V in draws:
            U, u, _ = train(V * size, u0, A, y, T=20000, stop_at=1e-8)
            P = learned_predictor(V * size, u0, A, y, sign=u)
            tr.append(trace_at_fit(U, u, A))
            te.append(excess_test_loss(P, Lam, Gam) / excess_test_loss(min_norm_fit(A, y), Lam, Gam))
        sharp[delta].append([size, np.median(tr), np.median(te)])
out["trace_vs_test"] = {str(k): v for k, v in sharp.items()}

fig, ax = plt.subplots(figsize=(6.6, 4.6))
for delta in deltas:
    c = np.array(sharp[delta])
    line, = ax.plot(c[:, 1], c[:, 2], "o-", label=f"$\\delta={delta:g}$")
    ax.annotate("", xy=(c[-1, 1], c[-1, 2]), xytext=(c[-2, 1], c[-2, 2]),
                arrowprops=dict(arrowstyle="->", color=line.get_color()))
ax.axhline(1, ls=":", c="gray")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set(xlabel="Hessian trace at the fit", ylabel="test loss / min-norm test loss",
       title="arrows point toward larger starts")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("figures/fig_sharpness.pdf")

# 5. letting the cross-terms train, test loss on 50k fresh prompts
Xt, Yt, At, yt = sample_prompts(50000, d, N, Lam, rng)


def train_full(X, Y, y, U0, u0, c0, frozen):
    _, C = full_predict(X, Y, N, U0, u0, c0[:d], c0[d:])
    xq = X[:, N]

    def rhs(t, z):
        U, u, c = z[:d * d].reshape(d, d), z[d * d], z[d * d + 1:]
        a = np.append(c[:d], u)
        W = np.vstack([U, c[d:]])
        r = ((C @ W) @ xq[:, :, None])[:, :, 0] @ a - y
        ga = r @ (C @ (W @ xq.T).T[:, :, None])[:, :, 0] / len(y)
        gW = (r[:, None] * (a @ C)).T @ xq / len(y)
        gc = np.zeros(2 * d) if frozen else np.append(ga[:d], gW[d])
        return -np.concatenate([gW[:d].ravel(), [ga[d]], gc])

    z0 = np.concatenate([U0.ravel(), [u0], c0])
    z = solve_ivp(rhs, (0, 6000), z0, method="LSODA", rtol=1e-9, atol=1e-11).y[:, -1]
    U, u, c = z[:d * d].reshape(d, d), z[d * d], z[d * d + 1:]
    pred, _ = full_predict(Xt, Yt, N, U, u, c[:d], c[d:])
    return 0.5 * np.mean((pred - yt) ** 2) - best_test, np.linalg.norm(c)


out["cross_terms"] = {}
for s in [0.1, 0.5, 1.0]:
    frozen, from0, rand, size0 = [], [], [], []
    for t in range(20):
        X, Y, A, y = sample_prompts(8, d, N, Lam, rng)
        U0 = rand_U(s, rng)
        c_rand = rng.standard_normal(2 * d) * s / np.sqrt(d)
        frozen.append(train_full(X, Y, y, U0, s, np.zeros(2 * d), True)[0])
        loss, c_size = train_full(X, Y, y, U0, s, np.zeros(2 * d), False)
        from0.append(loss)
        size0.append(c_size)
        rand.append(train_full(X, Y, y, U0, s, c_rand, False)[0])
    frozen, from0, rand = np.array(frozen), np.array(from0), np.array(rand)
    out["cross_terms"][str(s)] = {"frozen": np.median(frozen), "from_0": np.median(from0),
                                  "worse_from_0": int(np.sum(from0 > frozen)),
                                  "random": np.median(rand), "worse_random": int(np.sum(rand > frozen)),
                                  "cross_size_from_0": np.median(size0), "of": 20}

# 6. gradient descent with a finite step size vs the gradient flow prediction
out["finite_steps"] = {}
for u0, size in [(2.0, 1.0), (0.5, 0.5)]:
    for eta in [0.001, 0.01, 0.03, 0.06]:
        changes, ratios, diverged = [], [], 0
        for t in range(20):
            X, Y, A, y = sample_prompts(8, d, N, Lam, rng)
            U0 = rand_U(size, rng)
            U, u = U0.copy(), u0
            for k in range(400000):
                gU, gu = fit_grads(U, u, A, y)
                U, u = U - eta * gU, u - eta * gu
                loss = fit_loss(U, u, A, y)
                if loss < 1e-14 or not loss < 1e8:
                    break
            if not loss < 1e8:
                diverged += 1
            if loss < 1e-14:
                P_flow = learned_predictor(U0, u0, A, y, sign=u0)
                changes.append(u**2 - np.sum(U**2) - (u0**2 - size**2))
                ratios.append(excess_test_loss(u * U, Lam, Gam) / excess_test_loss(P_flow, Lam, Gam))
        out["finite_steps"][f"u0={u0}, |U0|={size}, eta={eta}"] = {
            "converged": len(changes), "diverged": diverged, "of": 20,
            "delta_change": np.median(changes) if changes else None,
            "test_ratio_vs_flow": np.median(ratios) if ratios else None}

json.dump(out, open("results/fewprompt.json", "w"), indent=1, default=float)
for k in ["formula_check", "sharpness", "cross_terms", "finite_steps"]:
    print(k, json.dumps(out[k], indent=1, default=float))
