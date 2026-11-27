# one layer linear attention from Zhang, Frei and Bartlett (2024)
# with the cross-terms at 0 the prediction is u * m^T U x_q = <A, P> with P = u U
import numpy as np
from scipy.integrate import solve_ivp


def make_problem(d, N, eigs, seed):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    Lam = Q @ np.diag(eigs) @ Q.T
    Gam = (1 + 1 / N) * Lam + np.trace(Lam) / N * np.eye(d)
    return Lam, Gam


# infinitely many prompts

def pop_loss(U, u, Lam, Gam):
    a = np.trace(Gam @ Lam @ U @ Lam @ U.T)
    b = np.trace(Lam @ Lam @ U.T)
    return 0.5 * u**2 * a - u * b


def pop_grads(U, u, Lam, Gam):
    gU = u**2 * Gam @ Lam @ U @ Lam - u * Lam @ Lam
    gu = u * np.trace(Gam @ Lam @ U @ Lam @ U.T) - np.trace(Lam @ Lam @ U.T)
    return gU, gu


def min_loss(Lam, Gam):
    return -0.5 * np.trace(Lam @ Lam @ np.linalg.inv(Gam))


def end_u2(delta, G):
    # positive root of s^2 - delta s - G^2 = 0
    return (delta + np.sqrt(delta**2 + 4 * G**2)) / 2


def excess_test_loss(P, Lam, Gam):
    E = P - np.linalg.inv(Gam)
    return 0.5 * np.trace(Gam @ Lam @ E @ Lam @ E.T)


# finitely many prompts

def sample_prompts(B, d, N, Lam, rng):
    w = rng.standard_normal((B, d))
    X = rng.standard_normal((B, N + 1, d)) @ np.linalg.cholesky(Lam).T  # last row is the query
    Y = (X @ w[:, :, None])[:, :, 0]
    m = (Y[:, None, :N] @ X[:, :N])[:, 0] / N
    A = m[:, :, None] * X[:, N][:, None, :]
    return X, Y, A, Y[:, N]


def fit_loss(U, u, A, y):
    r = u * np.sum(A * U, axis=(1, 2)) - y
    return 0.5 * np.mean(r**2)


def fit_grads(U, u, A, y):
    r = u * np.sum(A * U, axis=(1, 2)) - y
    gP = np.sum(r[:, None, None] * A, axis=0) / len(y)
    return u * gP, np.sum(gP * U)


def min_norm_fit(A, y):
    M = A.reshape(len(y), -1)
    return (M.T @ np.linalg.solve(M @ M.T, y)).reshape(A.shape[1:])


def learned_predictor(U0, u0, A, y, sign):
    # Theorem E, P = P_mn + u_end * U_unseen(0)
    B, d, _ = A.shape
    M = A.reshape(B, -1)
    if B >= d * d:
        return np.linalg.lstsq(M, y, rcond=None)[0].reshape(d, d)
    Q, _ = np.linalg.qr(M.T)
    U_seen = (Q @ (Q.T @ U0.ravel())).reshape(d, d)
    P_mn = min_norm_fit(A, y)
    delta_seen = u0**2 - np.sum(U_seen**2)
    u_end = np.sign(sign) * np.sqrt(end_u2(delta_seen, np.linalg.norm(P_mn)))
    return P_mn + u_end * (U0 - U_seen)


def flow(grads, U0, u0, T, stop=None):
    # gradient flow dU/dt = -gU, du/dt = -gu, stops early when stop(U, u) hits 0
    d = U0.shape[0]

    def rhs(t, z):
        gU, gu = grads(z[:-1].reshape(d, d), z[-1])
        return -np.append(gU.ravel(), gu)

    events = None
    if stop is not None:
        def events(t, z):
            return stop(z[:-1].reshape(d, d), z[-1])
        events.terminal = True

    sol = solve_ivp(rhs, (0, T), np.append(U0.ravel(), u0), method="LSODA",
                    rtol=1e-10, atol=1e-12, events=events, dense_output=True)
    return sol.y[:-1, -1].reshape(d, d), sol.y[-1, -1], sol


# full model with the cross-terms c_pv and c_kq

def full_predict(X, Y, N, U, u, c_pv, c_kq):
    B, _, d = X.shape
    E = np.zeros((B, d + 1, N + 1))
    E[:, :d] = X.transpose(0, 2, 1)
    E[:, d, :N] = Y[:, :N]
    C = E @ E.transpose(0, 2, 1) / N
    a = np.append(c_pv, u)
    W = np.vstack([U, c_kq])
    xq = X[:, N]
    pred = ((C @ W) @ xq[:, :, None])[:, :, 0] @ a
    return pred, C
