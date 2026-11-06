import json
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "DSL-StrongPasswordData.csv")
RESULTS = os.path.join(ROOT, "results")

GAMMA = 0.07
MU = 0.25
N_PCA = 20


def load_raw():
    df = pd.read_csv(DATA)
    subjects = sorted(df["subject"].unique())
    return df, subjects


def select_users(n_users):
    df, subjects = load_raw()
    enrolled = subjects[:n_users]
    unseen = subjects[n_users:]
    return df, enrolled, unseen


class Preprocessor:
    def __init__(self, n_pca=N_PCA):
        self.s1 = StandardScaler()
        self.pca = PCA(n_components=n_pca, random_state=0)
        self.poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        self.s2 = StandardScaler()

    def fit(self, x):
        z = self.poly.fit_transform(self.pca.fit_transform(self.s1.fit_transform(x)))
        self.s2.fit(z)
        return self

    def transform(self, x):
        z = self.poly.transform(self.pca.transform(self.s1.transform(x)))
        return self.s2.transform(z)


def feature_columns(df):
    return [c for c in df.columns if c not in ("subject", "sessionIndex", "rep")]


def make_split(n_users=10, seed=42, test_size=0.25):
    df, enrolled, unseen = select_users(n_users)
    cols = feature_columns(df)
    d = df[df["subject"].isin(enrolled)].reset_index(drop=True)
    label = {s: i for i, s in enumerate(enrolled)}
    y = d["subject"].map(label).to_numpy()
    x = d[cols].to_numpy(dtype=np.float64)
    idx = np.arange(len(y))
    tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    pre = Preprocessor().fit(x[tr])
    out = {
        "xtr": pre.transform(x[tr]),
        "ytr": y[tr],
        "xte": pre.transform(x[te]),
        "yte": y[te],
        "train_idx": tr,
        "test_idx": te,
        "enrolled": enrolled,
        "pre": pre,
        "k": n_users,
    }
    if unseen:
        u = df[df["subject"].isin(unseen)]
        out["xunseen"] = pre.transform(u[cols].to_numpy(dtype=np.float64))
        out["unseen_subject"] = u["subject"].to_numpy()
    return out


def add_bias(x):
    return np.hstack([x, np.ones((x.shape[0], 1))])


def one_hot(y, k):
    t = np.zeros((len(y), k))
    t[np.arange(len(y)), y] = 1.0
    return t


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def least_squares_poly(a, degree=3, grid=4001):
    z = np.linspace(-a, a, grid)
    s = sigmoid(z)
    v = np.vander(z, degree + 1, increasing=True)
    return np.linalg.lstsq(v, s, rcond=None)[0]


def minimax_poly(a, degree=3, iters=60):
    n = degree + 2
    nodes = a * np.cos(np.pi * np.arange(n)[::-1] / (n - 1))
    z = np.linspace(-a, a, 20001)
    coef = least_squares_poly(a, degree)
    for _ in range(iters):
        v = np.vander(nodes, degree + 1, increasing=True)
        m = np.hstack([v, ((-1.0) ** np.arange(n))[:, None]])
        sol = np.linalg.solve(m, sigmoid(nodes))
        coef = sol[:-1]
        err = np.polyval(coef[::-1], z) - sigmoid(z)
        sign = np.sign(err)
        sign[sign == 0] = 1
        breaks = np.where(np.diff(sign) != 0)[0]
        segments = np.split(np.arange(len(z)), breaks + 1)
        ext = np.array([seg[np.argmax(np.abs(err[seg]))] for seg in segments])
        if len(ext) < n:
            break
        if len(ext) > n:
            keep = np.argsort(-np.abs(err[ext]))[:n]
            ext = np.sort(ext[keep])
        new_nodes = z[ext]
        if np.allclose(new_nodes, nodes, atol=1e-9):
            break
        nodes = new_nodes
    return coef


class Activation:
    def __init__(self, kind="exact", mu=MU, a=8.0):
        self.kind = kind
        self.mu = mu
        self.a = a
        if kind == "ls3":
            self.coef = least_squares_poly(a, 3)
        elif kind == "minimax3":
            self.coef = minimax_poly(a, 3)
        else:
            self.coef = None

    def __call__(self, z):
        if self.kind == "exact":
            return sigmoid(z)
        if self.kind == "linear":
            return 0.5 + self.mu * z
        zc = np.clip(z, -self.a, self.a)
        return np.polyval(self.coef[::-1], zc)

    def depth(self):
        return {"exact": None, "linear": 1, "ls3": 2, "minimax3": 2}[self.kind]


def gradient(w, x, t, act):
    p = act(x @ w)
    return x.T @ (p - t) / x.shape[0]


BETA = 0.9
ROUNDS = 50


def train_centralized(xtr, ytr, k, rounds=ROUNDS, act=None, gamma=GAMMA, beta=BETA):
    act = act or Activation("exact")
    x = add_bias(xtr)
    t = one_hot(ytr, k)
    w = np.zeros((x.shape[1], k))
    v = np.zeros_like(w)
    for _ in range(rounds):
        v = beta * v + gamma * gradient(w, x, t, act)
        w = w - v
    return w


def partition_disjoint(ytr, k, users_per_client=2):
    clients = []
    for c in range(k // users_per_client):
        users = list(range(c * users_per_client, (c + 1) * users_per_client))
        clients.append(np.where(np.isin(ytr, users))[0])
    return clients


def partition_groups(ytr, k, n_clients=5):
    groups = np.array_split(np.arange(k), n_clients)
    return [np.where(np.isin(ytr, g))[0] for g in groups]


def partition_iid(ytr, n_clients, seed):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ytr))
    return [np.sort(p) for p in np.array_split(idx, n_clients)]


def partition_dirichlet(ytr, n_clients, alpha, seed, min_size=10):
    rng = np.random.default_rng(seed)
    k = ytr.max() + 1
    while True:
        parts = [[] for _ in range(n_clients)]
        for c in range(k):
            idx = rng.permutation(np.where(ytr == c)[0])
            prop = rng.dirichlet(alpha * np.ones(n_clients))
            cuts = (np.cumsum(prop) * len(idx)).astype(int)[:-1]
            for p, chunk in zip(parts, np.split(idx, cuts)):
                p.extend(chunk.tolist())
        if min(len(p) for p in parts) >= min_size:
            return [np.sort(np.array(p)) for p in parts]


def train_fedavg(xtr, ytr, k, clients, rounds=ROUNDS, act=None, gamma=GAMMA, beta=BETA,
                 local_steps=1, rule="fedavg", prox_mu=0.01, history=None, xte=None, yte=None,
                 transform=None, gate=None, norm_bound=None, norms=None):
    act = act or Activation("exact")
    x = add_bias(xtr)
    t = one_hot(ytr, k)
    w = np.zeros((x.shape[1], k))
    v = np.zeros_like(w)
    c_global = np.zeros_like(w)
    c_local = [np.zeros_like(w) for _ in clients]
    data = []
    for i, idx in enumerate(clients):
        xi, ti = x[idx], t[idx]
        if transform is not None and i in transform:
            xi, ti = transform[i](xi, ti)
        data.append((xi, ti))
    for r in range(rounds):
        deltas, sizes, new_c = [], [], []
        for i, (xi, ti) in enumerate(data):
            wi = w.copy()
            for _ in range(local_steps):
                g = gradient(wi, xi, ti, act)
                if rule == "fedprox":
                    g = g + prox_mu * (wi - w)
                if rule == "scaffold":
                    g = g - c_local[i] + c_global
                wi = wi - gamma * g
            if rule == "scaffold":
                new_c.append(c_local[i] - c_global + (w - wi) / (local_steps * gamma))
            deltas.append(w - wi)
            sizes.append(len(xi))
        if norms is not None:
            norms.append([float(np.linalg.norm(d)) for d in deltas])
        keep = list(range(len(deltas)))
        if norm_bound is not None:
            deltas = [d * min(1.0, norm_bound / max(np.linalg.norm(d), 1e-12)) for d in deltas]
        if gate is not None:
            keep = gate(w, v, deltas, sizes, keep)
        a = np.array([sizes[i] for i in keep], dtype=float)
        a = a / a.sum()
        delta = sum(ai * deltas[i] for ai, i in zip(a, keep))
        v = beta * v + delta
        w = w - v
        if rule == "scaffold":
            frac = len(clients)
            c_global = c_global + sum((new_c[i] - c_local[i]) for i in range(frac)) / frac
            c_local = new_c
        if history is not None and xte is not None:
            history.append(evaluate(w, xte, yte, k, act))
    return w


def scores(w, x):
    return add_bias(x) @ w


def predict(w, x):
    return np.argmax(scores(w, x), axis=1)


def evaluate(w, x, y, k, act=None):
    pred = predict(w, x)
    acc = float(np.mean(pred == y))
    recall = np.array([np.mean(pred[y == c] == c) for c in range(k)])
    prec = np.array([np.mean(y[pred == c] == c) if np.any(pred == c) else 0.0 for c in range(k)])
    out = {
        "accuracy": acc,
        "macro_recall": float(recall.mean()),
        "macro_precision": float(prec.mean()),
        "min_recall": float(recall.min()),
        "max_recall": float(recall.max()),
        "std_recall": float(recall.std()),
    }
    if act is not None:
        p = np.clip(act(scores(w, x)), 1e-7, 1 - 1e-7)
        t = one_hot(y, k)
        out["bce_loss"] = float(-np.mean(t * np.log(p) + (1 - t) * np.log(1 - p)))
    return out


def confusion(w, x, y, k):
    pred = predict(w, x)
    m = np.zeros((k, k), dtype=int)
    for a, b in zip(y, pred):
        m[a, b] += 1
    return m


def save_json(name, obj):
    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, name)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    return path


def mean_std(values):
    v = np.asarray(values, dtype=float)
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
