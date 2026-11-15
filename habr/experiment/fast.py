"""Тот же TD-поиск, скомпилированный numba: честное сравнение по времени.
ALT-оценка, множители и перцептрон считаются лениво, при первом обнаружении вершины."""

from __future__ import annotations

import heapq
import math
import time as clock

import numpy as np
from numba import njit

from core import DATA, DAYTIME, PEAK_EVENING, PEAK_MORNING, Graph, Landmarks

MODE_DIJKSTRA, MODE_WEIGHTED, MODE_HOURLY, MODE_NN, MODE_TABLE = 0, 1, 2, 3, 4


@njit(cache=True)
def _lb(fwd, bwd, v, t):
    best = 0.0
    for k in range(fwd.shape[0]):
        a = fwd[k, t] - fwd[k, v]
        b = bwd[k, v] - bwd[k, t]
        if a > best:
            best = a
        if b > best:
            best = b
    return best


@njit(cache=True)
def _mlp(feat, mean, std, w0, b0, w1, b1, w2, b2, h1, h2):
    nf = feat.shape[0]
    for j in range(h1.shape[0]):
        acc = b0[j]
        for i in range(nf):
            acc += (feat[i] - mean[i]) / std[i] * w0[i, j]
        h1[j] = acc if acc > 0.0 else 0.0
    for j in range(h2.shape[0]):
        acc = b1[j]
        for i in range(h1.shape[0]):
            acc += h1[i] * w1[i, j]
        h2[j] = acc if acc > 0.0 else 0.0
    out = b2[0]
    for i in range(h2.shape[0]):
        out += h2[i] * w2[i, 0]
    return out


@njit(cache=True)
def search(first, head, t0, am, ae, ad, pm_t, pe_t, pd_t, xs, ys, s, t, tau0,
           mode, fwd, bwd, weight, mult, alpha, mean, std, w0, b0, w1, b1, w2, b2, cell, table):
    n = first.shape[0] - 1
    dist = np.full(n, np.inf)
    hval = np.full(n, -1.0)
    closed = np.zeros(n, dtype=np.bool_)
    h1 = np.empty(w0.shape[1])
    h2 = np.empty(w1.shape[1])
    feat = np.empty(10)
    xt, yt = xs[t], ys[t]
    rt = math.sqrt(xt * xt + yt * yt)
    dist[s] = tau0
    heap = [(tau0, tau0, s)]
    expanded = 0
    evals = 0
    while len(heap) > 0:
        _, du, u = heapq.heappop(heap)
        if closed[u]:
            continue
        closed[u] = True
        expanded += 1
        if u == t:
            break
        m = (du % 86400.0) / 60.0
        i = int(m)
        f = m - i
        pm = pm_t[i] + (pm_t[i + 1] - pm_t[i]) * f
        pe = pe_t[i] + (pe_t[i + 1] - pe_t[i]) * f
        pd = pd_t[i] + (pd_t[i + 1] - pd_t[i]) * f
        for e in range(first[u], first[u + 1]):
            v = head[e]
            if closed[v]:
                continue
            dv = du + t0[e] * (1.0 + am[e] * pm + ae[e] * pe + ad[e] * pd)
            if dv < dist[v]:
                dist[v] = dv
                h = 0.0
                if mode != 0:
                    if hval[v] < 0.0:
                        lb = _lb(fwd, bwd, v, t)
                        if mode == 1:
                            h = lb * weight
                        elif mode == 2:
                            h = lb * mult[int((dv % 86400.0) // 900.0)]
                        elif mode == 4:
                            h = lb * math.exp(alpha * table[cell[v], cell[t], int((dv % 86400.0) // 900.0)])
                        else:
                            xv, yv = xs[v], ys[v]
                            phase = (dv % 86400.0) / 86400.0 * 2.0 * math.pi
                            feat[0] = xv / 17.0
                            feat[1] = yv / 17.0
                            feat[2] = xt / 17.0
                            feat[3] = yt / 17.0
                            feat[4] = math.sqrt(xv * xv + yv * yv) / 17.0
                            feat[5] = rt / 17.0
                            feat[6] = math.sqrt((xv - xt) ** 2 + (yv - yt) ** 2) / 30.0
                            feat[7] = math.log1p(lb / 60.0)
                            feat[8] = math.sin(phase)
                            feat[9] = math.cos(phase)
                            corr = _mlp(feat, mean, std, w0, b0, w1, b1, w2, b2, h1, h2)
                            evals += 1
                            if corr < 0.0:
                                corr = 0.0
                            h = lb * math.exp(alpha * corr)
                        hval[v] = h
                    else:
                        h = hval[v]
                heapq.heappush(heap, (dv + h, dv, v))
    return dist[t] - tau0, expanded, evals


class Fast:
    def __init__(self, g: Graph, lm: Landmarks):
        self.g = g
        self.args = (np.asarray(g.first, dtype=np.int64), np.asarray(g.head, dtype=np.int64), g.t0_arr, g.am_arr,
                     g.ae_arr, g.ad_arr, PEAK_MORNING, PEAK_EVENING, DAYTIME, g.x, g.y)
        self.fwd = np.ascontiguousarray(lm.fwd)
        self.bwd = np.ascontiguousarray(lm.bwd)
        self.empty = np.zeros(1)
        self.nets = {}
        self.tables = {}
        self.no_table = (np.zeros(g.n, dtype=np.int64), np.zeros((1, 1, 96)))

    def net(self, name: str):
        if name not in self.nets:
            d = np.load(DATA / f"{name}.npz")
            self.nets[name] = tuple(np.ascontiguousarray(d[k]) for k in ("mean", "std", "w0", "b0", "w1", "b1", "w2", "b2"))
        return self.nets[name]

    def table(self, name):
        if name not in self.tables:
            d = np.load(DATA / f"{name}.npz")
            self.tables[name] = (d["cell"], np.ascontiguousarray(d["table"]))
        return self.tables[name]

    def run(self, s, t, tau, mode=0, weight=1.0, mult=None, alpha=0.0, net="mlp", table=None):
        params = self.net(net)
        tab = self.table(table) if table else self.no_table
        mult = mult if mult is not None else np.ones(96)
        start = clock.perf_counter()
        travel, expanded, evals = search(*self.args, s, t, tau, mode, self.fwd, self.bwd, weight, mult, alpha, *params, *tab)
        return travel, expanded, evals, clock.perf_counter() - start
