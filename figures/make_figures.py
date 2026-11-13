import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("SOURCE_DATE_EPOCH", "1788998400")
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e3", "grid.linewidth": 0.5,
    "axes.axisbelow": True,
})
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUT = "#0b0b0b", "#52514e"
CS = Path(os.environ.get("DECODE_EVIDENCE", Path(__file__).resolve().parents[1] / "evidence"))
OUT = Path(__file__).resolve().parents[1] / "outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def jload(rel):
    return json.loads((CS / rel).read_text())


def fold_mean(v):
    return float(np.mean(v))


def fig_board():
    fam = {
        "GBM-lags": (67.78, 84.40),
        "GBM-flat": (65.59, 80.65),
    }
    left_only = {"GBM-cash": (66.32, "D", -0.09), "LLM-agent": (64.74, "^", -0.18)}
    right_only = {"CE-Seq": 82.28}
    myo_bound, ceiling = 76.13, 100.0

    grpA = [("full disclosure", 9.35, (8.68, 10.02)),
            ("exact-response step", 15.61, (14.14, 17.16)),
            ("filter-feature step", -0.68, (-1.60, 0.22))]
    grpB = [("contact discrimination", 6.68, (5.73, 7.77)),
            ("contact propagation", 3.26, (2.31, 4.01)),
            ("organic (spontaneous)", -0.32, (-0.63, -0.05))]

    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 1.48),
                               gridspec_kw={"width_ratios": [1, 1.15]})
    cols = {"GBM-lags": BLUE, "GBM-flat": AQUA}
    lab_y = {"GBM-lags": 89.0, "CE-Seq": 82.3, "GBM-flat": 72.5}
    lab_c = {"GBM-lags": INK, "CE-Seq": ORANGE, "GBM-flat": INK}

    def label(name, yv):
        a.annotate(name, xy=(1.03, yv), xytext=(1.16, lab_y[name]),
                   color=lab_c[name], fontsize=6.5, va="center", ha="left",
                   arrowprops=dict(arrowstyle="-", color=MUT, lw=0.5,
                                   shrinkA=0, shrinkB=1.5))
    for name, (m, s) in fam.items():
        a.plot([0, 1], [m, s], "-o", color=cols[name], lw=1.5, ms=4)
        label(name, s)
    for name, (m, mk, dx) in left_only.items():
        a.plot([dx], [m], mk, color=MUT, ms=3.5)
    for name, s in right_only.items():
        a.plot([1], [s], "o", color=ORANGE, ms=4, zorder=3)
        label(name, s)
    a.axhline(ceiling, color=MUT, lw=0.8, ls="--")
    a.annotate("exact optimum", (1.82, ceiling), color=MUT, fontsize=6,
               va="bottom", ha="right")
    a.axhline(myo_bound, color=MUT, lw=0.8, ls=":")
    from matplotlib.lines import Line2D
    hs = [Line2D([], [], color=MUT, marker="D", ls="none", ms=3.5, label="GBM-cash"),
          Line2D([], [], color=MUT, marker="^", ls="none", ms=3.5, label="LLM-agent"),
          Line2D([], [], color=MUT, lw=0.8, ls=":", label="1-month optimum")]
    a.legend(handles=hs, loc="upper left", bbox_to_anchor=(-0.02, 0.93),
             ncol=2, fontsize=5.5, frameon=False, handlelength=1.2,
             handletextpad=0.4, columnspacing=0.8, borderaxespad=0.2)
    a.set_xlim(-0.25, 1.85)
    a.set_xticks([0, 1], ["1-month\nhorizon", "6-month\nhorizon"])
    a.set_ylim(60, 103)
    a.set_ylabel("% of exact optimum")
    a.set_title("(a) Horizon vs solver class", loc="left", color=INK)

    rows = grpA
    ys = [2.0, 1.0, 0.0]
    for yi, (lab, v, ci) in zip(ys, rows):
        b.plot([ci[0], ci[1]], [yi, yi], "-", color=BLUE, lw=1.2)
        b.plot([v], [yi], "o", color=BLUE, ms=5)
    b.axvline(0, color=MUT, lw=0.8)
    b.set_yticks(ys, [r[0] for r in rows])
    b.set_ylim(-0.6, 2.6)
    b.set_xlabel("disclosure value (points)")
    b.set_title("(b) Disclosure values", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "f2_board.pdf")
    plt.close(fig)


def fig_dial():
    d = [0.0, 0.5, 1.0, 2.0, 4.0]
    flat = [0.4018, 0.4353, 0.4254, 0.4181, 0.3380]
    hist = [0.5283, 0.6101, 0.5802, 0.5593, 0.4521]
    orac = [0.8961, 0.8693, 0.8775, 0.8850, 0.8822]
    g = jload("entering_state/geometry_sweep/geometry_sweep_results.json")
    anchors = {0.0: 0.1265, 0.5: 0.1748, 4.0: 0.1141}
    dd3 = [0.0, 0.5, 4.0]
    draws = {"anchor": [anchors[x] for x in dd3]}
    for gs in (25001, 25002, 25003, 25004):
        draws[str(gs)] = [g[f"g{gs}_d{str(x).replace('.', 'p')}"]
                          ["separation"] for x in dd3]

    fig, (a, b) = plt.subplots(1, 2, figsize=(4.6, 1.42),
                               gridspec_kw={"width_ratios": [1.25, 1]})
    series = [("snapshot", flat, ORANGE, "o", "-"),
              ("history", hist, BLUE, "s", "-"),
              ("true state", orac, VIOLET, "^", "--")]
    for lab, ysv, c, mk, ls in series:
        a.plot(d, ysv, ls, marker=mk, color=c, lw=1.5, ms=4, label=lab)
    a.set_xlabel(r"memory parameter $\delta$")
    a.set_ylabel(r"$R^2$")
    a.set_ylim(0.30, 1.0)
    a.set_xticks(d)
    a.annotate("true state", (2.0, 0.925), color=VIOLET,
               fontsize=6.5, ha="center", va="bottom")
    a.annotate("history", (0.55, 0.640), color=BLUE,
               fontsize=6.5, va="bottom")
    a.annotate("snapshot", (0.9, 0.385), color=ORANGE,
               fontsize=6.5, ha="left", va="top")
    a.set_title("(a) Information sets", loc="left", color=INK)

    xpos = [0, 1, 2]
    for name, ysv in draws.items():
        if name == "anchor":
            b.plot(xpos, ysv, "-o", color=BLUE, lw=1.6, ms=4, zorder=3,
                   label="members' geometry")
        else:
            b.plot(xpos, ysv, "-o", color=MUT, lw=0.8, ms=2.5, alpha=0.8,
                   label="four further draws" if name == "25001" else None)
    b.axhline(0, color=MUT, lw=0.8)
    b.set_xticks(xpos, ["0", "0.5", "4"])
    b.set_xlim(-0.2, 2.2)
    b.set_ylim(-0.02, 0.62)
    b.legend(loc="upper left", bbox_to_anchor=(0.0, 1.15), fontsize=5.5, frameon=False,
             handlelength=1.4, labelspacing=0.2, borderaxespad=0.2)
    b.set_xlabel(r"memory parameter $\delta$")
    b.set_ylabel("separation")
    b.set_title("(b) Separation per draw", loc="left", color=INK)
    fig.tight_layout(w_pad=2.0)
    fig.savefig(OUT / "f3_dial.pdf")
    plt.close(fig)


def fig_arch():
    lags = {w: fold_mean(jload(f"entering_state/memory_probes/archsweep_{w}.json")["lags"])
            for w in ("v7", "d05", "d40")}

    def arch_delta(w, key, files):
        vals = []
        for f in files:
            data = jload(f"entering_state/memory_probes/{f}")
            for k, v in data.items():
                if k.startswith(f"{key}_seed") and not k.endswith("_stops") \
                        and isinstance(v, list):
                    vals.append(fold_mean(v))
        return float(np.mean(vals)) - lags[w]

    worlds = ["v7", "d05", "d40"]
    labels = ["Base", "Mem-0.5", "Mem-4"]
    learned = {}
    for key in ("gru128x2", "lstm128x2", "gru32long"):
        learned[key] = [arch_delta(w, key, [f"archsweep_{w}.json"]
                                   + (["archsweep_d40_seeds234.json"]
                                      if w == "d40" else []))
                        for w in worlds]
    anchor = []
    for w, files in (("v7", ["bdemo_v7train_control.json"]),
                     ("d05", ["bdemo_d05train.json"]),
                     ("d40", ["bdemo_d40train.json",
                              "archsweep_d40_seeds234.json"])):
        vals = []
        for f in files:
            data = jload(f"entering_state/memory_probes/{f}")
            for k, v in data.items():
                if k.startswith("gru_seed") and not k.endswith("_stops") \
                        and isinstance(v, list):
                    vals.append(fold_mean(v))
        anchor.append(float(np.mean(vals)) - lags[w])
    learned["gru32"] = anchor
    filt, prem = [], []
    for w, f in (("v7", "bdemo_v7train_control.json"),
                 ("d05", "bdemo_d05train.json"),
                 ("d40", "bdemo_d40train.json")):
        data = jload(f"entering_state/memory_probes/{f}")
        filt.append(fold_mean(data["filter"]) - lags[w])
        ba = data.get("bayes_analytic")
        if ba is None:
            ba = jload("entering_state/memory_probes/bdemo_v7train_audit.json")["bayes_analytic"]
        prem.append(float(ba) - lags[w])

    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(3.15, 1.9))
    ax.plot(x, prem, "--", color=VIOLET, lw=1.5, marker="D", ms=4,
            label="information reference (no fit)")
    ax.plot(x, filt, "-", color=BLUE, lw=1.5, marker="o", ms=4,
            label="reference filter (fitted)")
    mks = {"gru32": "v", "gru32long": "^", "lstm128x2": "s", "gru128x2": "P"}
    names = {"gru128x2": "GRU, large", "lstm128x2": "LSTM, large",
             "gru32long": "GRU, small, long budget", "gru32": "GRU, small"}
    for key in ("gru128x2", "lstm128x2", "gru32long", "gru32"):
        ax.plot(x, learned[key], "-", color=MUT, lw=0.9, marker=mks[key],
                ms=3.5, alpha=0.85, label=names[key])
    ax.axhline(0, color=MUT, lw=0.8)
    ax.annotate("history = 0", (1.5, -0.012), color=MUT, fontsize=6.5,
                ha="center", va="top")
    ax.set_ylim(-0.06, 0.53)
    ax.set_yticks([0.0, 0.1, 0.2, 0.3])
    ax.set_xticks(x, labels)
    ax.set_ylabel("$\\Delta R^2$")
    ax.set_title("(a) Memory value", loc="left", color=INK)
    ax.legend(loc="upper left", ncol=2, fontsize=5.5, frameon=True,
              framealpha=1.0, edgecolor="none", borderpad=0.2,
              handlelength=1.4, labelspacing=0.2, columnspacing=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "f5_arch.pdf")
    plt.close(fig)


def fig_reobs():
    q1 = jload("reobservation/v8_curve_q01.json")
    q2 = jload("reobservation/v8_curve_q02.json")

    def to_pp(q):
        c = q["CL_oracle"]
        return (100 * q["reobs_mean"] / c,
                100 * q["reobs_ci"][0] / c, 100 * q["reobs_ci"][1] / c)

    m1, lo1, hi1 = to_pp(q1)
    m2, lo2, hi2 = to_pp(q2)
    xs = [0.0, 0.1, 0.2]
    mean = [0.04, m1, m2]
    lo = [0.04 - (-0.14), m1 - lo1, m2 - lo2]
    hi = [0.24 - 0.04, hi1 - m1, hi2 - m2]
    fig, ax = plt.subplots(figsize=(2.2, 1.9))
    ax.errorbar(xs, mean, yerr=[lo, hi], fmt="-o", color=BLUE, lw=1.5,
                ms=5, capsize=3)
    for xv, mv in zip(xs, mean):
        off = (9, -5) if xv == 0.1 else (8, 5)
        ax.annotate(f"{mv:+.2f}", (xv, mv), textcoords="offset points",
                    xytext=off, color=INK, fontsize=7)
    ax.set_xlabel(r"process noise $\sigma$")
    ax.set_ylabel("% of upper\nreference")
    ax.set_title("(b) Re-observation gain", loc="left", color=INK)
    ax.set_xticks(xs)
    fig.tight_layout()
    fig.savefig(OUT / "f6_reobs.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_board()
    fig_dial()
    fig_arch()
    fig_reobs()
    for f in sorted(OUT.glob("*.pdf")):
        print(f.name, f.stat().st_size)
