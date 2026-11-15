from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def cover() -> None:
    hours = np.linspace(0, 24, 600)
    free_flow = 4.0
    minutes = free_flow * (1 + peak(hours, 8.5, 1.2, 1.4) + peak(hours, 18.5, 1.5, 1.1))
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(hours, minutes, color=BLUE, linewidth=4)
    ax.axhline(free_flow, color=GREY, linestyle="--", linewidth=2)
    ax.fill_between(hours, free_flow, minutes, color=BLUE, alpha=0.15)
    ax.set_xlim(0, 24)
    ax.set_ylim(2.8, 11)
    ax.set_xticks([0, 6, 12, 18, 24], ["00:00", "06:00", "12:00", "18:00", "24:00"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.text(8.5, 10.1, "8:30", ha="center", color=BLUE, fontsize=16)
    ax.text(18.5, 8.9, "18:30", ha="center", color=BLUE, fontsize=16)
    ax.text(2.5, 4.35, "ночью", ha="center", color=GREY, fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "00-cover.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    travel_time_profile()
    cover()
