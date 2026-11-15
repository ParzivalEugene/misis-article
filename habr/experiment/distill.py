"""Перегнать нейросеть в таблицу поправок: район вершины × район цели × 15 минут."""

import sys

import numpy as np
from scipy.cluster.vq import kmeans2

from core import DATA, MLP, build_landmarks, features, load_graph
from fast import _lb

K = int(sys.argv[1]) if len(sys.argv) > 1 else 64
NET = sys.argv[2] if len(sys.argv) > 2 else "mlp"


def main() -> None:
    g = load_graph()
    lm = build_landmarks(g)
    model = MLP(DATA / f"{NET}.npz")
    pts = np.column_stack([g.x, g.y])
    centroids, cell = kmeans2(pts, K, seed=1, minit="++")
    reps = np.array([int(np.argmin(((pts - c) ** 2).sum(1))) for c in centroids])
    cv, ct = np.meshgrid(np.arange(K), np.arange(K), indexing="ij")
    cv, ct = cv.ravel(), ct.ravel()
    v, t = reps[cv], reps[ct]
    lb = np.array([_lb(lm.fwd, lm.bwd, a, b) for a, b in zip(v, t)])
    lb = np.maximum(lb, 60.0)
    table = np.zeros((K, K, 96))
    for b in range(96):
        tau = b * 900 + 450
        x = features(g, v, t, tau, lb)
        table[cv, ct, b] = np.maximum(model(x), 0.0)
    np.savez(DATA / f"table{K}.npz", cell=cell.astype(np.int64), table=table)
    print("table", table.shape, "mean mult", np.exp(table).mean())


if __name__ == "__main__":
    main()
