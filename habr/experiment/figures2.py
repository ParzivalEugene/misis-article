"""Дополнительные картинки: карта пробок, что выучила сеть, таблица за сутки, выигрыш по длине, GIF поиска."""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import PillowWriter
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, Normalize

from core import DATA, MLP, alt_bound, build_landmarks, edge_time_array, features, load_graph, make_nn, td_search
from figures import BLUE, GREEN, IMG, ORANGE, PEAK, gmean, load

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def segments(g):
    return np.stack([np.column_stack([g.x[g.u_arr], g.y[g.u_arr]]), np.column_stack([g.x[g.v_arr], g.y[g.v_arr]])], axis=1)


def map_axes(ax, g):
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_xlim(g.x.min(), g.x.max())
    ax.set_ylim(g.y.min(), g.y.max())


def traffic_map(g) -> None:
    speed = g.length / edge_time_array(g, 8.5 * 3600) * 3.6
    order = np.argsort(-speed)  # медленные рисуем поверх
    seg = segments(g)[order]
    fig, ax = plt.subplots(figsize=(8, 7.4))
    lc = LineCollection(seg, array=speed[order], cmap="RdYlGn", norm=Normalize(8, 45), linewidths=0.9)
    ax.add_collection(lc)
    map_axes(ax, g)
    cb = fig.colorbar(lc, ax=ax, shrink=0.7, pad=0.02)
    cb.set_label("скорость в 8:30, км/ч")
    fig.tight_layout()
    fig.savefig(IMG / "02-traffic.png", dpi=150)
    plt.close(fig)


def learned_map(g, lm, model) -> None:
    from core import alt_bound_from
    k = int(np.argmin(g.x ** 2 + g.y ** 2))  # перекрёсток у Кремля
    verts = np.arange(g.n)
    tau = 8.5 * 3600
    lb_in = alt_bound(lm, k)  # от каждого перекрёстка до Кремля
    lb_out = alt_bound_from(lm, k)  # от Кремля до каждого перекрёстка
    mult_in = np.exp(np.maximum(model(features(g, verts, k, tau, lb_in)), 0))
    mult_out = np.exp(np.maximum(model(features(g, k, verts, tau, lb_out)), 0))
    seg = segments(g)
    norm = Normalize(1.0, 3.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2))
    for ax, (title, mult, lb) in zip(axes, [("в центр", mult_in, lb_in), ("из центра", mult_out, lb_out)]):
        edge = (mult[g.u_arr] + mult[g.v_arr]) / 2
        ok = (lb[g.u_arr] > 60) & (lb[g.v_arr] > 60)
        ax.add_collection(LineCollection(seg[~ok], colors="#e6e6e6", linewidths=0.5))
        lc = LineCollection(seg[ok], array=edge[ok], cmap="magma_r", norm=norm, linewidths=0.9)
        ax.add_collection(lc)
        ax.plot(g.x[k], g.y[k], "*", color=BLUE, markersize=16, markeredgecolor="white")
        map_axes(ax, g)
        ax.set_title(f"8:30, поездки {title}", pad=8)
        print(title, np.quantile(mult[lb > 60], [0.05, 0.5, 0.95]).round(2))
    cb = fig.colorbar(lc, ax=axes, shrink=0.75, pad=0.02)
    cb.set_label("во сколько раз дорога длиннее\nоценки ALT, по мнению сети")
    fig.savefig(IMG / "05-learned.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def table_curve() -> None:
    hourly = np.exp(np.load(DATA / "hourly.npy"))
    hours = (np.arange(96) + 0.5) / 4
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.step(hours, hourly, where="mid", color=GREEN, linewidth=2.5)
    ax.set_xlim(0, 24)
    ax.set_ylim(1, max(hourly) * 1.12)
    ax.set_xticks(range(0, 25, 3), [f"{h}:00" for h in range(0, 25, 3)])
    ax.set_ylabel("множитель к оценке")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(IMG / "04-table.png", dpi=160)
    plt.close(fig)


def by_length() -> None:
    df = load()
    d = df[df.slot.isin(PEAK)]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for name, color, label in [("A* (ALT)", BLUE, "обычный A*"), ("почасовой множитель", GREEN, "таблица по времени суток"),
                               ("нейросеть mlp8x8", ORANGE, "нейросеть 8×8")]:
        x = d[(d.method == name) & (d.param == 1.0)].groupby("rank").speedup.apply(gmean)
        ax.plot(2 ** x.index, x.values, "o-", color=color, linewidth=2.5, label=label)
    ranks = [2 ** r for r in range(6, 15, 2)]
    ax.set_xscale("log", base=2)
    ax.set_xticks(ranks, [f"{r:,}".replace(",", " ") for r in ranks])
    ax.minorticks_off()
    ax.set_xlabel("номер цели в порядке обхода Дейкстры: чем правее, тем длиннее поездка")
    ax.set_ylabel("перебор меньше,\nчем у Дейкстры, раз")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(IMG / "07-by-length.png", dpi=160)
    plt.close(fig)


def search_gif(g, lm, model) -> None:
    s = int(np.argmin((g.x + 11) ** 2 + (g.y - 9) ** 2))
    t = int(np.argmin((g.x - 10) ** 2 + (g.y + 9) ** 2))
    tau = 8.5 * 3600
    lb = alt_bound(lm, t)
    _, _, od = td_search(g, s, t, tau, None, record_order=True)
    _, _, on = td_search(g, s, t, tau, make_nn(g, lb, t, model, 1.0), record_order=True)
    od, on = np.array(od), np.array(on)
    frames = 48
    step = int(np.ceil(len(od) / frames))
    seg = segments(g)
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2))
    dots, titles = [], []
    for ax, (name, color) in zip(axes, [("Дейкстра", BLUE), ("A* с нейросетевой поправкой", ORANGE)]):
        ax.add_collection(LineCollection(seg, colors="#dddddd", linewidths=0.35))
        dots.append(ax.scatter([], [], s=2.2, color=color, linewidths=0))
        ax.plot(g.x[s], g.y[s], "o", color="black", markersize=7)
        ax.plot(g.x[t], g.y[t], "*", color="black", markersize=13)
        map_axes(ax, g)
        titles.append(ax.set_title(name, fontsize=13))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.01, wspace=0.04)
    writer = PillowWriter(fps=12)
    with writer.saving(fig, str(IMG / "09-search.gif"), dpi=72):
        for f in list(range(1, frames + 1)) + [frames] * 18:
            k = min(f * step, len(od))
            for dot, order, title, name in zip(dots, [od, on], titles, ["Дейкстра", "A* с поправкой"]):
                shown = order[: min(k, len(order))]
                dot.set_offsets(np.column_stack([g.x[shown], g.y[shown]]))
                title.set_text(f"{name}: {len(shown):,} перекрёстков".replace(",", " "))
            writer.grab_frame()
    plt.close(fig)


if __name__ == "__main__":
    g = load_graph()
    lm = build_landmarks(g)
    model = MLP(DATA / "mlp8x8.npz")
    which = sys.argv[1:] or ["traffic", "learned", "table", "length", "gif"]
    if "traffic" in which:
        traffic_map(g)
    if "learned" in which:
        learned_map(g, lm, model)
    if "table" in which:
        table_curve()
    if "length" in which:
        by_length()
    if "gif" in which:
        search_gif(g, lm, model)
