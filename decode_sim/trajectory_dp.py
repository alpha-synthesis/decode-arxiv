from __future__ import annotations

import itertools

import numpy as np


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def solve_loan(
    logits: np.ndarray,
    beta: float,
    decay: float,
    fat_add: np.ndarray,
    fatigue_applies: np.ndarray,
    cash: np.ndarray,
    costs: np.ndarray,
    q: np.ndarray,
    d_grid: np.ndarray | None = None,
) -> dict:
    logits = np.asarray(logits, dtype=float)
    H, C, D = logits.shape
    fat_add = np.asarray(fat_add, dtype=float)
    fatigue_applies = np.asarray(fatigue_applies, dtype=bool)
    cash = np.asarray(cash, dtype=float)
    costs = np.asarray(costs, dtype=float)
    q = np.asarray(q, dtype=float)

    best_value, best_path = -np.inf, None
    for path in itertools.product(range(C), repeat=H):
        fatigue = np.empty(H)
        f = 0.0
        for t, c in enumerate(path):
            fatigue[t] = f
            f = decay * f + fat_add[c]
        value_next = 0.0
        d_choice = [0] * H
        for t in range(H - 1, -1, -1):
            c = path[t]
            pen = beta * fatigue[t] if fatigue_applies[c] else 0.0
            p = _sigmoid(logits[t, c, :] - pen)
            total = -costs[c] + p * cash[t, :] + (1.0 - p) * (1.0 - q[t]) * value_next
            k = int(np.argmax(total))
            d_choice[t] = k
            value_next = float(total[k])
        if value_next > best_value:
            best_value = value_next
            best_path = tuple((path[t], d_choice[t]) for t in range(H))

    f, survival, myopic_value = 0.0, 1.0, 0.0
    myopic_path = []
    for t in range(H):
        best_imm, pick = -np.inf, None
        for c in range(C):
            pen = beta * f if fatigue_applies[c] else 0.0
            p = _sigmoid(logits[t, c, :] - pen)
            immediate = p * cash[t, :] - costs[c]
            k = int(np.argmax(immediate))
            if immediate[k] > best_imm:
                best_imm = float(immediate[k])
                pick = (c, k, float(p[k]))
        c, k, p_pick = pick
        myopic_value += survival * best_imm
        myopic_path.append((c, k))
        survival *= (1.0 - p_pick) * (1.0 - q[t])
        f = decay * f + fat_add[c]

    return {
        "ceiling": float(best_value),
        "path": best_path,
        "myopic": float(myopic_value),
        "myopic_path": tuple(myopic_path),
    }
