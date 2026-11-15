"""Граф, профили загрузки, ALT и TD-поиск (Дейкстра и варианты A*)."""

from __future__ import annotations

import heapq
import math
import time as clock
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra as static_dijkstra

DATA = Path(__file__).resolve().parent / "data"
CENTER = (55.7539, 37.6208)
DAY = 86400.0
CLASSES = ["motorway", "trunk", "primary", "secondary", "tertiary", "residential", "other"]
FREE_SPEED = np.array([90, 80, 60, 50, 40, 30, 25], dtype=float)  # км/ч ночью
# Насколько класс дороги проседает в пик. Подобрано так, чтобы МКАД в утренний пик
# ехал ~37 км/ч при 90+ ночью, а Садовое ~27 км/ч (данные «Яндекса»).
AMPLITUDE = np.array([3.0, 3.0, 2.5, 2.0, 1.4, 0.7, 0.5])
MINUTES = 1440


def gauss(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    d = np.minimum(np.abs(x - mu), 24 - np.abs(x - mu))
    return np.exp(-0.5 * (d / sigma) ** 2)


def smooth_step(x: np.ndarray, a: float, b: float, width: float) -> np.ndarray:
    return 1 / (1 + np.exp(-(x - a) / width)) * (1 / (1 + np.exp((x - b) / width)))


HOURS = np.arange(MINUTES + 1) / 60.0
PEAK_MORNING = gauss(HOURS, 8.5, 1.1)
PEAK_EVENING = gauss(HOURS, 18.7, 1.4)
DAYTIME = 0.45 * smooth_step(HOURS, 9.5, 20.5, 0.6)


@dataclass
class Graph:
    n: int
    lat: np.ndarray
    lon: np.ndarray
    x: np.ndarray  # км от центра на восток
    y: np.ndarray  # км от центра на север
    first: list  # CSR: рёбра вершины u — с first[u] по first[u+1]
    head: list
    t0: list  # время свободного проезда, с
    am: list  # множители пиков по ребру
    ae: list
    ad: list
    t0_arr: np.ndarray
    am_arr: np.ndarray
    ae_arr: np.ndarray
    ad_arr: np.ndarray
    u_arr: np.ndarray
    v_arr: np.ndarray
    cls: np.ndarray
    length: np.ndarray


def load_graph() -> Graph:
    d = np.load(DATA / "graph.npz")
    lat, lon = d["lat"], d["lon"]
    u, v, length, cls = d["u"], d["v"], d["length"], d["cls"].astype(int)
    order = np.lexsort((v, u))
    u, v, length, cls = u[order], v[order], length[order], cls[order]
    n = len(lat)
    kx = 111.32 * math.cos(math.radians(CENTER[0]))
    x = (lon - CENTER[1]) * kx
    y = (lat - CENTER[0]) * 110.57
    t0 = length / (FREE_SPEED[cls] / 3.6)
    mx, my = (x[u] + x[v]) / 2, (y[u] + y[v]) / 2
    r = np.hypot(mx, my)
    radial = 0.6 + 0.8 * np.exp(-r / 7.0)  # в центре плотнее
    dx, dy = x[v] - x[u], y[v] - y[u]
    seg = np.hypot(dx, dy) + 1e-9
    inward = (-(mx * dx + my * dy)) / (seg * (r + 1e-9))  # cos угла к центру
    inbound, outbound = np.clip(inward, 0, 1), np.clip(-inward, 0, 1)
    base = AMPLITUDE[cls] * radial
    am = base * (0.4 + 0.6 * inbound)  # утром тянутся в центр
    ae = base * (0.4 + 0.6 * outbound)  # вечером из центра
    ad = base * 0.5
    first = np.zeros(n + 1, dtype=np.int64)
    np.add.at(first, u + 1, 1)
    first = np.cumsum(first)
    return Graph(n, lat, lon, x, y, first.tolist(), v.tolist(), t0.tolist(), am.tolist(), ae.tolist(), ad.tolist(),
                 t0, am, ae, ad, u, v, cls, length)


def profile_tables() -> tuple[list, list, list]:
    return PEAK_MORNING.tolist(), PEAK_EVENING.tolist(), DAYTIME.tolist()


def edge_time_array(g: Graph, tau: float) -> np.ndarray:
    m = (tau % DAY) / 60.0
    i = int(m)
    f = m - i
    pm = PEAK_MORNING[i] * (1 - f) + PEAK_MORNING[i + 1] * f
    pe = PEAK_EVENING[i] * (1 - f) + PEAK_EVENING[i + 1] * f
    pd = DAYTIME[i] * (1 - f) + DAYTIME[i + 1] * f
    return g.t0_arr * (1 + g.am_arr * pm + g.ae_arr * pe + g.ad_arr * pd)


# ---------- ALT ----------

@dataclass
class Landmarks:
    fwd: np.ndarray  # d(L, v), с
    bwd: np.ndarray  # d(v, L), с


def build_landmarks(g: Graph, k: int = 16, seed: int = 0) -> Landmarks:
    path = DATA / f"landmarks{k}.npz"
    if path.exists():
        d = np.load(path)
        return Landmarks(d["fwd"], d["bwd"])
    mat = csr_matrix((g.t0_arr, (g.u_arr, g.v_arr)), shape=(g.n, g.n))
    rng = np.random.default_rng(seed)
    chosen = [int(rng.integers(g.n))]
    dist_sum = static_dijkstra(mat, indices=chosen[0])
    chosen = [int(np.argmax(dist_sum))]
    best = static_dijkstra(mat, indices=chosen[0])
    while len(chosen) < k:  # farthest-point
        nxt = int(np.argmax(best))
        chosen.append(nxt)
        best = np.minimum(best, static_dijkstra(mat, indices=nxt))
    fwd = static_dijkstra(mat, indices=chosen)
    bwd = static_dijkstra(mat.T.tocsr(), indices=chosen)
    np.savez(path, fwd=fwd, bwd=bwd, ids=np.array(chosen))
    return Landmarks(fwd, bwd)


def alt_bound(lm: Landmarks, t: int) -> np.ndarray:
    """Нижняя оценка времени от каждой вершины до t по свободному движению."""
    a = lm.fwd[:, t][:, None] - lm.fwd
    b = lm.bwd - lm.bwd[:, t][:, None]
    return np.maximum(0.0, np.maximum(a, b).max(axis=0))


def alt_bound_from(lm: Landmarks, s: int) -> np.ndarray:
    """Нижняя оценка времени от s до каждой вершины."""
    a = lm.fwd - lm.fwd[:, s][:, None]
    b = lm.bwd[:, s][:, None] - lm.bwd
    return np.maximum(0.0, np.maximum(a, b).max(axis=0))


# ---------- поиск ----------

@dataclass
class Result:
    arrival: float
    expanded: int
    seconds: float
    h_seconds: float
    h_calls: int
    settled: list | None = None


def td_search(g: Graph, s: int, t: int | None, tau0: float, heuristic=None, keep_settled: bool = False,
              record_order: bool = False):
    """TD-Дейкстра / A*. heuristic(vertices, times) -> оценки (батч на одно раскрытие).
    Повторно вершины не раскрываются. При t=None — полный обход (метки для всех вершин)."""
    pm_t, pe_t, pd_t = PEAK_MORNING_L, PEAK_EVENING_L, DAYTIME_L
    first, head, t0, am, ae, ad = g.first, g.head, g.t0, g.am, g.ae, g.ad
    INF = math.inf
    dist = {s: tau0}
    hval = {}
    closed = set()
    h_time = 0.0
    h_calls = 0
    start = clock.perf_counter()
    if heuristic is not None:
        h0 = clock.perf_counter()
        hval[s] = heuristic([s], [tau0])[0]
        h_time += clock.perf_counter() - h0
        h_calls += 1
    heap = [(tau0 + hval.get(s, 0.0), tau0, s)]
    order = [] if record_order else None
    expanded = 0
    push, pop = heapq.heappush, heapq.heappop
    while heap:
        _, du, u = pop(heap)
        if u in closed:
            continue
        closed.add(u)
        expanded += 1
        if record_order:
            order.append(u)
        if u == t:
            break
        m = (du % 86400.0) / 60.0
        i = int(m)
        f = m - i
        pm = pm_t[i] + (pm_t[i + 1] - pm_t[i]) * f
        pe = pe_t[i] + (pe_t[i + 1] - pe_t[i]) * f
        pd = pd_t[i] + (pd_t[i + 1] - pd_t[i]) * f
        new_v = []
        new_d = []
        for e in range(first[u], first[u + 1]):
            v = head[e]
            if v in closed:
                continue
            dv = du + t0[e] * (1.0 + am[e] * pm + ae[e] * pe + ad[e] * pd)
            if dv < dist.get(v, INF):
                dist[v] = dv
                if heuristic is None:
                    push(heap, (dv, dv, v))
                elif v in hval:
                    push(heap, (dv + hval[v], dv, v))
                else:
                    new_v.append(v)
                    new_d.append(dv)
        if new_v:
            h0 = clock.perf_counter()
            hs = heuristic(new_v, new_d)
            h_time += clock.perf_counter() - h0
            h_calls += 1
            for v, dv, hv in zip(new_v, new_d, hs):
                hval[v] = hv
                push(heap, (dv + hv, dv, v))
    seconds = clock.perf_counter() - start
    arrival = dist.get(t, INF) if t is not None else INF
    res = Result(arrival - tau0 if t is not None else INF, expanded, seconds, h_time, h_calls,
                 list(closed) if keep_settled else None)
    if t is None or record_order:
        return res, dist, order
    return res


PEAK_MORNING_L, PEAK_EVENING_L, DAYTIME_L = profile_tables()


# ---------- эвристики ----------

def make_lb(lb: np.ndarray, weight: float = 1.0):
    lbl = (lb * weight).tolist()
    return lambda vs, ts: [lbl[v] for v in vs]


def make_hourly(lb: np.ndarray, hourly_log: np.ndarray, alpha: float):
    lbl = lb.tolist()
    mult = np.exp(alpha * hourly_log).tolist()  # 96 корзин по 15 минут
    return lambda vs, ts: [lbl[v] * mult[int((tt % 86400.0) // 900)] for v, tt in zip(vs, ts)]


FEATURES = 10


def features(g: Graph, vs, t, ts, lb_v) -> np.ndarray:
    """Признаки пары (v, t) в момент ts. v и t — номер или массив."""
    vs = np.asarray(vs)
    t = np.asarray(t)
    n = max(vs.size, t.size)
    xv, yv = np.broadcast_to(g.x[vs], n), np.broadcast_to(g.y[vs], n)
    xt, yt = np.broadcast_to(g.x[t], n), np.broadcast_to(g.y[t], n)
    phase = (np.broadcast_to(np.asarray(ts, dtype=float), n) % DAY) / DAY * 2 * np.pi
    return np.column_stack([
        xv / 17, yv / 17, xt / 17, yt / 17,
        np.hypot(xv, yv) / 17, np.hypot(xt, yt) / 17,
        np.hypot(xv - xt, yv - yt) / 30,
        np.log1p(np.broadcast_to(lb_v, n) / 60.0),
        np.sin(phase), np.cos(phase),
    ])


class MLP:
    def __init__(self, path: Path):
        d = np.load(path)
        self.mean, self.std = d["mean"], d["std"]
        self.w = [d[f"w{i}"] for i in range(3)]
        self.b = [d[f"b{i}"] for i in range(3)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        h = (x - self.mean) / self.std
        h = np.maximum(h @ self.w[0] + self.b[0], 0)
        h = np.maximum(h @ self.w[1] + self.b[1], 0)
        return (h @ self.w[2] + self.b[2]).ravel()


def make_nn(g: Graph, lb: np.ndarray, t: int, model: MLP, alpha: float):
    def h(vs, ts):
        vs = np.asarray(vs)
        lb_v = lb[vs]
        corr = model(features(g, vs, t, np.asarray(ts), lb_v))
        return (lb_v * np.exp(alpha * np.maximum(corr, 0.0))).tolist()
    return h
