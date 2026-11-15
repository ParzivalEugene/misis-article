"""Графики для статьи из results_fast.csv и карта области поиска."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from core import DATA, MLP, alt_bound, build_landmarks, load_graph, make_hourly, make_lb, make_nn, td_search

IMG = Path(__file__).resolve().parent.parent / "img"
BLUE, ORANGE, GREEN, GREY, RED = "#2f6db5", "#e07b39", "#3a9a5b", "#8a8a8a", "#c0392b"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})
NET = sys.argv[1] if len(sys.argv) > 1 else "mlp8x8"
PEAK = ["утро 08:30", "вечер 18:30"]


def gmean(x):
    return float(np.exp(np.log(x).mean()))


def load() -> pd.DataFrame:
    df = pd.read_csv(DATA / "results_fast.csv")
    base = df[df.method == "Дейкстра"][["slot", "s", "t", "expanded", "us"]].rename(columns={"expanded": "exp_d", "us": "us_d"})
    df = df.merge(base, on=["slot", "s", "t"])
    df["speedup"] = df.exp_d / df.expanded
    df["tspeedup"] = df.us_d / df.us
    return df


def tradeoff(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (title, slots, xmax) in zip(axes, [("Час пик (8:30 и 18:30)", PEAK, 4), ("Ночь (3:00)", ["ночь 03:00"], 8)]):
        d = df[df.slot.isin(slots)]
        alt = gmean(d[d.method == "A* (ALT)"].speedup)
        ax.scatter([0], [alt], color="black", zorder=5, s=60)
        ax.annotate("обычный A*", (0, alt), xytext=(xmax * 0.18, alt - 0.3 if alt < 5 else alt - 2.5),
                    arrowprops={"arrowstyle": "->", "color": GREY}, va="center")
        for name, color, label in [("взвешенный A*", GREY, "постоянный вес"), ("почасовой множитель", GREEN, "множитель по времени суток"),
                                   (f"нейросеть {NET}", ORANGE, "нейросетевая поправка")]:
            g = d[d.method == name].groupby("param").agg(sp=("speedup", gmean), err=("err", "mean")).reset_index()
            ax.plot(np.r_[0, g.err * 100], np.r_[alt, g.sp], "o-", color=color, linewidth=2.5, label=label, markersize=5)
        ax.set_xlim(-0.05 * xmax, xmax)
        ax.set_xlabel("средняя ошибка времени маршрута, %")
        ax.set_title(title)
    axes[0].set_ylabel("во сколько раз меньше\nраскрытых вершин, чем у Дейкстры")
    axes[0].legend(frameon=False, loc="lower right", fontsize=10, bbox_to_anchor=(1.0, 0.08))
    fig.tight_layout()
    fig.savefig(IMG / "03-tradeoff.png", dpi=160)
    plt.close(fig)


def by_slot(df: pd.DataFrame) -> None:
    slots = ["ночь 03:00", "утро 08:30", "день 13:00", "вечер 18:30"]
    methods = [("A* (ALT)", 1.0, BLUE, "обычный A*"), ("почасовой множитель", 1.0, GREEN, "множитель по времени суток"),
               (f"нейросеть {NET}", 1.0, ORANGE, "нейросетевая поправка")]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    width = 0.26
    for k, (name, param, color, label) in enumerate(methods):
        vals = [gmean(df[(df.slot == s) & (df.method == name) & (df.param == param)].speedup) for s in slots]
        bars = ax.bar(np.arange(4) + (k - 1) * width, vals, width, color=color, label=label)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}", ha="center", fontsize=9)
    ax.set_xticks(range(4), slots)
    ax.set_ylabel("во сколько раз меньше\nраскрытий, чем у Дейкстры")
    ax.set_ylim(0, 17.5)
    ax.legend(frameon=False, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    fig.savefig(IMG / "02-by-slot.png", dpi=160)
    plt.close(fig)


def timing(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.slot.isin(PEAK)]
    rows = []
    for name, param in [("A* (ALT)", 1.0), ("взвешенный A*", 1.5), ("почасовой множитель", 1.0),
                        ("нейросеть mlp8x8", 1.0), ("нейросеть mlp16x16", 1.0), ("нейросеть mlp32x32", 1.0), ("нейросеть mlp", 1.0)]:
        x = d[(d.method == name) & (d.param == param)]
        rows.append(dict(method=name, param=param, exp=gmean(x.speedup), time=gmean(x.tspeedup), err=x.err.mean() * 100,
                         p95=x.err.quantile(0.95) * 100, opt=(x.err < 1e-9).mean() * 100, us=x.us.median()))
    return pd.DataFrame(rows)


def search_map() -> None:
    g = load_graph()
    lm = build_landmarks(g)
    model = MLP(DATA / f"{NET}.npz")
    hourly = np.load(DATA / "hourly.npy")
    # длинный запрос через центр: с северо-запада на юго-восток
    s = int(np.argmin((g.x + 11) ** 2 + (g.y - 9) ** 2))
    t = int(np.argmin((g.x - 10) ** 2 + (g.y + 9) ** 2))
    tau = 8.5 * 3600
    lb = alt_bound(lm, t)
    panels = [("Дейкстра", None), ("обычный A*", make_lb(lb)), ("нейросетевая поправка", make_nn(g, lb, t, model, 1.0))]
    seg = np.stack([np.column_stack([g.x[g.u_arr], g.y[g.u_arr]]), np.column_stack([g.x[g.v_arr], g.y[g.v_arr]])], axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.6))
    for ax, (title, h) in zip(axes, panels):
        res = td_search(g, s, t, tau, h, keep_settled=True)
        ax.add_collection(LineCollection(seg, colors="#d9d9d9", linewidths=0.4))
        st = np.array(res.settled)
        ax.scatter(g.x[st], g.y[st], s=3, color=ORANGE if h is not None else BLUE, alpha=0.8, linewidths=0)
        ax.plot(g.x[s], g.y[s], "o", color="black", markersize=8)
        ax.plot(g.x[t], g.y[t], "*", color="black", markersize=14)
        ax.set_title(f"{title}\n{res.expanded:,} вершин".replace(",", " "))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_xlim(g.x.min(), g.x.max())
        ax.set_ylim(g.y.min(), g.y.max())
        print(title, res.expanded, round(res.arrival / 60, 1))
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(IMG / "04-map.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    if len(sys.argv) > 2 and sys.argv[2] == "map":
        search_map()
        sys.exit()
    df = load()
    tradeoff(df)
    by_slot(df)
    print(timing(df).round(2).to_string())
    search_map()
