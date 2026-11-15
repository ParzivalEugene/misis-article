"""Финальный эксперимент на скомпилированном поиске: раскрытия, время, ошибка."""

import time

import numpy as np
import pandas as pd

from core import DATA, build_landmarks, load_graph, td_search
from fast import MODE_DIJKSTRA, MODE_HOURLY, MODE_NN, MODE_TABLE, MODE_WEIGHTED, Fast

SLOTS = {"ночь 03:00": 3.0, "утро 08:30": 8.5, "день 13:00": 13.0, "вечер 18:30": 18.5}
SOURCES_PER_SLOT = 50
RANKS = range(6, 15)
WEIGHTS = [1.1, 1.2, 1.35, 1.5, 2.0, 3.0]
ALPHAS = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
NETS = ["mlp8x8", "mlp16x16", "mlp32x32", "mlp"]
TABLES = ["table32", "table64", "table128"]
REPEAT = 3


def main() -> None:
    g = load_graph()
    lm = build_landmarks(g)
    fast = Fast(g, lm)
    hourly = np.load(DATA / "hourly.npy")
    for net in NETS:
        fast.run(0, 1, 0.0, MODE_NN, alpha=1.0, net=net)
    rng = np.random.default_rng(2026)
    variants = [("Дейкстра", 0.0, dict(mode=MODE_DIJKSTRA)), ("A* (ALT)", 1.0, dict(mode=MODE_WEIGHTED, weight=1.0))]
    variants += [("взвешенный A*", w, dict(mode=MODE_WEIGHTED, weight=w)) for w in WEIGHTS]
    variants += [("почасовой множитель", a, dict(mode=MODE_HOURLY, mult=np.exp(a * hourly))) for a in ALPHAS]
    variants += [(f"нейросеть {net}", a, dict(mode=MODE_NN, alpha=a, net=net)) for net in NETS for a in ALPHAS]
    variants += [(f"таблица {tab}", a, dict(mode=MODE_TABLE, alpha=a, table=tab)) for tab in TABLES for a in ALPHAS]
    fast.run(0, 1, 0.0, MODE_TABLE, alpha=1.0, table=TABLES[0])
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
                t = int(order[2 ** r])
                opt = None
                for name, param, kw in variants:
                    best = np.inf
                    for _ in range(REPEAT):
                        travel, expanded, evals, sec = fast.run(s, t, tau, **kw)
                        best = min(best, sec)
                    if opt is None:
                        opt = travel
                    rows.append(dict(slot=slot, hour=hour, s=s, t=t, rank=r, method=name, param=param,
                                     expanded=expanded, evals=evals, us=best * 1e6, travel=travel, opt=opt))
        print(slot, f"{time.time() - start:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df["err"] = df.travel / df.opt - 1
    df.to_csv(DATA / "results_fast.csv", index=False)
    print("rows", len(df))


if __name__ == "__main__":
    main()
