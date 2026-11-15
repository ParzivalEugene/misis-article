"""Обучающая выборка из полных запусков TD-Дейкстры и обучение перцептрона."""

import time

import numpy as np
from sklearn.neural_network import MLPRegressor

from core import DATA, DAY, alt_bound_from, build_landmarks, features, load_graph, td_search

SOURCES, PER_SOURCE, BINS = 2500, 240, 8


def main(hidden=(64, 64), name="mlp") -> None:
    g = load_graph()
    lm = build_landmarks(g)
    rng = np.random.default_rng(42)
    rows, ys, groups, taus = [], [], [], []
    start = time.time()
    cache = DATA / "samples.npz"
    for k in range(0 if not cache.exists() else SOURCES, SOURCES):
        s = int(rng.integers(g.n))
        tau0 = float(rng.uniform(0, DAY))
        _, dist, _ = td_search(g, s, None, tau0)
        targets = np.fromiter(dist.keys(), dtype=np.int64)
        true = np.fromiter(dist.values(), dtype=float) - tau0
        lb = alt_bound_from(lm, s)[targets]
        ok = lb > 30
        targets, true, lb = targets[ok], true[ok], lb[ok]
        # поровну коротких и длинных пар: равномерно по корзинам log(lb)
        edges = np.quantile(np.log(lb), np.linspace(0, 1, BINS + 1))
        bins = np.clip(np.searchsorted(edges, np.log(lb), side="right") - 1, 0, BINS - 1)
        pick = np.concatenate([rng.choice(np.flatnonzero(bins == b), size=min(PER_SOURCE // BINS, int((bins == b).sum())), replace=False) for b in range(BINS)])
        rows.append(features(g, s, targets[pick], tau0, lb[pick]))
        ys.append(np.log(true[pick] / lb[pick]))
        groups.append(np.full(len(pick), k))
        taus.append(np.full(len(pick), tau0))
        if k % 500 == 0:
            print(k, f"{time.time() - start:.0f}s", flush=True)
    if cache.exists():
        d = np.load(cache)
        x, y, grp, tau = d["x"], d["y"], d["grp"], d["tau"]
    else:
        x, y, grp, tau = np.vstack(rows), np.concatenate(ys), np.concatenate(groups), np.concatenate(taus)
        np.savez(cache, x=x, y=y, grp=grp, tau=tau)
    print("samples", len(y), "label time", f"{time.time() - start:.0f}s", "mean ratio", np.exp(y).mean())
    val = grp >= int(SOURCES * 0.85)  # источники проверки не пересекаются с обучением
    mean, std = x[~val].mean(0), x[~val].std(0) + 1e-9
    model = MLPRegressor(hidden_layer_sizes=hidden, activation="relu", batch_size=1024, learning_rate_init=2e-3,
                         max_iter=60, early_stopping=True, n_iter_no_change=6, random_state=0, verbose=False)
    t = time.time()
    model.fit((x[~val] - mean) / std, y[~val])
    print("train", f"{time.time() - t:.0f}s", "iters", model.n_iter_)
    pred = model.predict((x[val] - mean) / std)
    rmse = float(np.sqrt(np.mean((pred - y[val]) ** 2)))
    base = float(np.sqrt(np.mean((y[val] - y[~val].mean()) ** 2)))
    # почасовой множитель: среднее log-отношение в 15-минутной корзине
    bin_ = (tau // 900).astype(int)
    hourly = np.array([y[~val][bin_[~val] == b].mean() for b in range(96)])
    hpred = hourly[bin_[val]]
    hrmse = float(np.sqrt(np.mean((hpred - y[val]) ** 2)))
    print(f"val RMSE log-ratio: nn {rmse:.4f}  hourly {hrmse:.4f}  const {base:.4f}")
    print("val median |rel err| nn", float(np.median(np.abs(np.exp(pred - y[val]) - 1))), "hourly", float(np.median(np.abs(np.exp(hpred - y[val]) - 1))))
    np.savez(DATA / f"{name}.npz", mean=mean, std=std, w0=model.coefs_[0], w1=model.coefs_[1], w2=model.coefs_[2],
             b0=model.intercepts_[0], b1=model.intercepts_[1], b2=model.intercepts_[2])
    np.save(DATA / "hourly.npy", hourly)
    np.savez(DATA / f"{name}_stats.npz", rmse=rmse, hrmse=hrmse, base=base, n=len(y), sources=SOURCES)


if __name__ == "__main__":
    import sys
    sizes = tuple(int(v) for v in sys.argv[1].split("x")) if len(sys.argv) > 1 else (64, 64)
    main(sizes, "mlp" if sizes == (64, 64) else "mlp" + "x".join(map(str, sizes)))
