from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Ellipse

OUT = Path(__file__).resolve().parent
BLUE = "#2f6db5"
ORANGE = "#e07b39"
GREEN = "#3a9a5b"
GREY = "#8a8a8a"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def peak(hours: np.ndarray, center: float, width: float, height: float) -> np.ndarray:
    return height * np.exp(-0.5 * ((hours - center) / width) ** 2)


def travel_time_profile() -> None:
    hours = np.linspace(0, 24, 600)
    free_flow = 4.0
    minutes = free_flow * (1 + peak(hours, 8.5, 1.2, 1.4) + peak(hours, 18.5, 1.5, 1.1))
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hours, minutes, color=BLUE, linewidth=2.5, label="время проезда дуги")
    ax.axhline(free_flow, color=GREY, linestyle="--", label="свободное движение (нижняя оценка)")
    ax.fill_between(hours, free_flow, minutes, color=BLUE, alpha=0.12)
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 12.5)
    ax.set_xticks(range(0, 25, 3))
    ax.set_xlabel("момент въезда, ч")
    ax.set_ylabel("время проезда, мин")
    ax.set_title("Один и тот же участок дороги в разное время суток")
    ax.annotate("утренний пик:\nв 2,4 раза дольше", xy=(9.0, 9.4), xytext=(11.2, 10.4), arrowprops={"arrowstyle": "->", "color": GREY})
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "01-profile.png", dpi=160)
    plt.close(fig)


def search_areas() -> None:
    fig, ax = plt.subplots(figsize=(8, 6.2))
    start, goal = np.array([1.5, 2.3]), np.array([6.5, 2.3])
    ax.add_patch(Circle(start, 5.0, color=GREY, alpha=0.15, label="Дейкстра: во все стороны"))
    ax.add_patch(Ellipse((4.0, 2.3), 6.9, 3.4, color=BLUE, alpha=0.22, label="A* с нижней оценкой в час пик"))
    ax.add_patch(Ellipse((4.0, 2.3), 5.9, 1.4, color=ORANGE, alpha=0.35, label="A* с уточнённой оценкой"))
    ax.plot(*start, "o", color="black", markersize=9)
    ax.plot(*goal, "*", color="black", markersize=16)
    ax.text(start[0] - 0.2, start[1] - 0.45, "старт", ha="center")
    ax.text(goal[0] + 0.1, goal[1] - 0.5, "цель", ha="center")
    ax.set_xlim(-3.7, 9)
    ax.set_ylim(-2.9, 7.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Чем точнее оценка оставшегося пути, тем уже область поиска (схема)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.16), ncol=1, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "02-search-area.png", dpi=160)
    plt.close(fig)


def alpha_knob() -> None:
    alphas = np.linspace(0, 1, 200)
    lower_bound = 10.0
    true_time = 16.0
    prediction = 17.5
    heuristic = lower_bound * np.exp(alphas * np.log(prediction / lower_bound))
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(alphas, heuristic, color=ORANGE, linewidth=2.5, label="оценка h_α")
    ax.axhline(true_time, color=GREEN, linestyle="--", label="настоящее оставшееся время")
    ax.axhline(lower_bound, color=GREY, linestyle=":", label="нижняя оценка (свободное движение)")
    crossing = float(np.log(true_time / lower_bound) / np.log(prediction / lower_bound))
    ax.axvspan(0, crossing, color=GREEN, alpha=0.08)
    ax.axvspan(crossing, 1, color=ORANGE, alpha=0.08)
    ax.text(crossing / 2, 18.2, "не завышает:\nмаршрут оптимален", ha="center", fontsize=10)
    ax.text((crossing + 1) / 2, 18.2, "завышает:\nбыстрее, но\nвозможна ошибка", ha="center", fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(8, 20.5)
    ax.set_xlabel("α — доля нейросетевой поправки")
    ax.set_ylabel("оценка, мин")
    ax.set_title("Ручка α на одной паре вершин (иллюстрация)")
    ax.legend(loc="lower right", frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "03-alpha.png", dpi=160)
    plt.close(fig)


def labeling() -> None:
    rng = np.random.default_rng(7)
    points = rng.uniform(0, 10, size=(160, 2))
    source = np.array([5.0, 5.0])
    distance = np.linalg.norm(points - source, axis=1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    scatter = ax.scatter(points[:, 0], points[:, 1], c=distance, cmap="viridis", s=28)
    targets = rng.choice(len(points), size=12, replace=False)
    ax.scatter(points[targets, 0], points[targets, 1], facecolors="none", edgecolors=ORANGE, s=140, linewidths=2, label="отобранные цели")
    ax.plot(*source, "o", color="red", markersize=12, label="вершина v, момент τ")
    fig.colorbar(scatter, ax=ax, label="точное время в пути из v")
    ax.set_title("Один запуск TD-Dijkstra размечает весь граф")
    ax.axis("off")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "04-labeling.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    travel_time_profile()
    search_areas()
    alpha_knob()
    labeling()
