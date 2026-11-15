"""Основной эксперимент: запросы по рангу Дейкстры, все варианты эвристик."""

import time

import numpy as np
import pandas as pd

from core import DATA, MLP, alt_bound, build_landmarks, load_graph, make_hourly, make_lb, make_nn, td_search

SLOTS = {"ночь 03:00": 3.0, "утро 08:30": 8.5, "день 13:00": 13.0, "вечер 18:30": 18.5}
SOURCES_PER_SLOT = 50
RANKS = range(6, 15)
WEIGHTS = [1.1, 1.2, 1.35, 1.5, 2.0, 3.0]
ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25]


def main() -> None:
    g = load_graph()
    lm = build_landmarks(g)
    model = MLP(DATA / "mlp.npz")
    hourly = np.load(DATA / "hourly.npy")
    rng = np.random.default_rng(2026)
    rows = []
    start = time.time()
    for slot, hour in SLOTS.items():
        tau = hour * 3600
        for _ in range(SOURCES_PER_SLOT):
            s = int(rng.integers(g.n))
            _, _, order = td_search(g, s, None, tau, record_order=True)
            for r in RANKS:
                if 2 ** r >= len(order):
                    continue
                t = order[2 ** r]
                lb = alt_bound(lm, t)
                variants = [("Дейкстра", 0.0, None), ("A* (ALT)", 1.0, make_lb(lb))]
                variants += [("взвешенный A*", w, make_lb(lb, w)) for w in WEIGHTS]
                variants += [("почасовой множитель", a, make_hourly(lb, hourly, a)) for a in ALPHAS]
                variants += [("нейросеть", a, make_nn(g, lb, t, model, a)) for a in ALPHAS]
                opt = None
                for name, param, h in variants:
                    res = td_search(g, s, t, tau, h)
                    if opt is None:
                        opt = res.arrival
                    rows.append(dict(slot=slot, hour=hour, s=s, t=t, rank=r, method=name, param=param,
                                     expanded=res.expanded, ms=res.seconds * 1000, h_ms=res.h_seconds * 1000,
                                     h_calls=res.h_calls, travel=res.arrival, opt=opt))
        print(slot, f"{time.time() - start:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["err"] = df.travel / df.opt - 1
    df.to_csv(DATA / "results.csv", index=False)
    print("rows", len(df))


if __name__ == "__main__":
    main()
