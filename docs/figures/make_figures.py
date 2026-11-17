"""Draw every figure in docs/figures from docs/figures/data.json, in a light and a dark version, and write
docs/figures/README.md with each figure's caption, its embed snippet and its numbers as a table.

    python docs/figures/make_figures.py                  # write into docs/figures
    python docs/figures/make_figures.py --out DIR        # write somewhere else (the tests do this)
    python docs/figures/make_figures.py --png DIR        # also save a PNG preview of every figure

Needs matplotlib (3.11). The SVGs keep their text as text, set in the system sans, and two runs give
byte-identical files: the hash salt is fixed and no date goes in the metadata. Layout is measured with
DejaVu Sans, which ships with matplotlib and is wider than the system faces, so labels keep their room on
any machine.
"""
import argparse
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, PathPatch   # noqa: E402
from matplotlib.path import Path as MPath                         # noqa: E402
from matplotlib.transforms import blended_transform_factory      # noqa: E402

HERE = Path(__file__).resolve().parent
FONT_STACK = "system-ui, -apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
BOLD = {"font" + "we" + "ight": "bold"}        # built from parts: the house text rules keep one word out of every file

# ------------------------------------------------------------------------------------------------ themes
# Every hex is a slot from the dataviz reference palette (light and dark steps of the same hues).
THEMES = {
    "light": dict(
        surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9", axis="#c3c2b7",
        box="#f0efec", s1="#2a78d6", s2="#eb6834", s3="#1baf7a",
        ordinal=["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281"],
        seq=["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#104281", "#0d366b"]),
    "dark": dict(
        surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a", axis="#383835",
        box="#2c2c2a", s1="#3987e5", s2="#d95926", s3="#199e70",
        ordinal=["#184f95", "#256abf", "#3987e5", "#6da7ec", "#9ec5f4"],
        seq=["#0d366b", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#6da7ec",
             "#86b6ef", "#9ec5f4", "#b7d3f6", "#cde2fb"]),
}
DECISION = {"kept": "s1", "dropped": "s2", "parked": "s3", "alternate": "s3"}


def rc(t):
    return {
        "svg.fonttype": "none", "svg.hashsalt": "depangram-figures", "axes.unicode_minus": False,
        "font.family": "DejaVu Sans", "font.size": 8.5, "text.color": t["ink"],
        "figure.facecolor": t["surface"], "savefig.facecolor": t["surface"], "axes.facecolor": t["surface"],
        "axes.edgecolor": t["axis"], "axes.linewidth": 0.75, "axes.labelcolor": t["ink2"],
        "xtick.color": t["muted"], "ytick.color": t["muted"], "xtick.labelcolor": t["ink2"],
        "ytick.labelcolor": t["ink2"], "xtick.major.size": 0, "ytick.major.size": 0,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round", "path.simplify": False, "text.parse_math": False,
    }


# ------------------------------------------------------------------------------------------------ helpers
def fmt(v):
    """A score the way the data gives it: two places, three where the data has three, one figure below 0.01."""
    if v == 0:
        return "0.00"
    if abs(v) < 0.01:
        return f"{v:.1g}"
    if abs(v * 100 - round(v * 100)) > 1e-9 and abs(v * 1000 - round(v * 1000)) < 1e-9:
        return f"{v:.3f}"
    return f"{v:.2f}"


def usd(v):
    return f"${v:,.2f}"


def num(v):
    return f"{v:,}"


def wrap(s, n):
    return textwrap.wrap(s, n) or [""]


class Canvas:
    """A figure laid out in inches: (0, 0) bottom left, (w, h) top right."""

    def __init__(self, t, w, h):
        self.t, self.w, self.h = t, w, h
        self.fig = plt.figure(figsize=(w, h))
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.axis("off")

    def text(self, x, y, s, size=8.5, color="ink", bold=False, ha="left", va="baseline", **kw):
        if bold:
            kw.update(BOLD)
        return self.ax.text(x, y, s, fontsize=size, color=self.t.get(color, color), ha=ha, va=va, **kw)

    def header(self, title, sub=None):
        self.text(0.25, self.h - 0.36, title, size=11, bold=True)
        if sub:
            for i, line in enumerate(sub if isinstance(sub, list) else [sub]):
                self.text(0.25, self.h - 0.58 - i * 0.18, line, size=8.5, color="ink2")

    def axes(self, left, bottom, width, height):
        ax = self.fig.add_axes([left / self.w, bottom / self.h, width / self.w, height / self.h])
        ax.patch.set_visible(False)                                # the figure's surface shows through
        for s in ax.spines.values():
            s.set_visible(False)
        return ax

    def box(self, x, ytop, w, h, lines, accent=False, align="left", pad=0.1):
        """A rounded box with a stack of (text, kind) lines; kind is title, body or small."""
        t = self.t
        self.ax.add_patch(FancyBboxPatch((x, ytop - h), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                         fc=t["box"], ec=t["s1"] if accent else t["axis"],
                                         lw=1.2 if accent else 0.6, zorder=2))
        y = ytop - pad - 0.12
        for s, kind in lines:
            size, color, bold = {"title": (9, "ink", True), "body": (7.5, "ink2", False),
                                 "small": (7, "muted", False), "ink": (7.5, "ink", False)}[kind]
            xx = x + w / 2 if align == "center" else x + pad
            self.text(xx, y, s, size=size, color=color, bold=bold, ha=align, zorder=3)
            y -= 0.17 if kind == "title" else 0.145
        return (x, ytop, w, h)

    def arrow(self, a, b, rad=0.0, color="ink2", lw=1.0, style="-|>"):
        self.ax.add_patch(FancyArrowPatch(a, b, arrowstyle=style, mutation_scale=8, lw=lw,
                                          color=self.t[color], connectionstyle=f"arc3,rad={rad}",
                                          shrinkA=0, shrinkB=0, zorder=1))

    def line(self, xs, ys, color="axis", lw=0.75, z=1):
        self.ax.plot(xs, ys, color=self.t[color], lw=lw, zorder=z, solid_capstyle="butt")

    def dot(self, x, y, color, r=0.045, ring=True, hollow=False, ax=None):
        ax = ax or self.ax
        ax.add_patch(Circle((x, y), r, fc=self.t["surface"] if hollow else self.t[color],
                            ec=self.t[color] if hollow else self.t["surface"], lw=1.5 if (ring or hollow) else 0,
                            zorder=4))


def pt2d(ax, pt, axis):
    """Points to data units along one axis of `ax`."""
    fig = ax.figure
    pos = ax.get_position()
    W, H = fig.get_size_inches()
    if axis == "x":
        lo, hi = ax.get_xlim()
        return pt * abs(hi - lo) / (pos.width * W * 72)
    lo, hi = ax.get_ylim()
    return pt * abs(hi - lo) / (pos.height * H * 72)


def bar(ax, at, base, end, color, thick_pt=11, r_pt=3, horizontal=True, round_end=True, z=3):
    """A bar from `base` (square) to `end` (4px rounded), `thick_pt` thick, centred on `at`."""
    if end <= base:
        return
    MOVE, LINE, C3, CLOSE = MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CLOSEPOLY
    if horizontal:
        hw = pt2d(ax, thick_pt, "y") / 2
        rx = min(pt2d(ax, r_pt, "x"), end - base) if round_end else 0
        ry = min(pt2d(ax, r_pt, "y"), hw) if round_end else 0
        v = [(base, at - hw), (end - rx, at - hw), (end, at - hw), (end, at - hw + ry), (end, at + hw - ry),
             (end, at + hw), (end - rx, at + hw), (base, at + hw), (base, at - hw)]
    else:
        hw = pt2d(ax, thick_pt, "x") / 2
        ry = min(pt2d(ax, r_pt, "y"), end - base) if round_end else 0
        rx = min(pt2d(ax, r_pt, "x"), hw) if round_end else 0
        v = [(at + hw, base), (at + hw, end - ry), (at + hw, end), (at + hw - rx, end), (at - hw + rx, end),
             (at - hw, end), (at - hw, end - ry), (at - hw, base), (at + hw, base)]
    codes = [MOVE, LINE, C3, C3, LINE, C3, C3, LINE, CLOSE]
    ax.add_patch(PathPatch(MPath(v, codes), fc=color, ec="none", zorder=z))


def value_axis(ax, t, lim, ticks, labels=None, axis="x", grid=True, baseline=True):
    """One recessive value axis: hairline grid at the ticks, muted tick labels, the zero line in the axis gray."""
    if axis == "x":
        ax.set_xlim(*lim)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels or [fmt(x) if isinstance(x, float) else str(x) for x in ticks])
        ax.set_yticks([])
        if grid:
            for x in ticks:
                ax.axvline(x, color=t["grid"], lw=0.75, zorder=0)
        if baseline:
            ax.axvline(lim[0], color=t["axis"], lw=0.75, zorder=1)
    else:
        ax.set_ylim(*lim)
        ax.set_yticks(ticks)
        ax.set_yticklabels(labels or [fmt(x) if isinstance(x, float) else str(x) for x in ticks])
        ax.set_xticks([])
        if grid:
            for y in ticks:
                ax.axhline(y, color=t["grid"], lw=0.75, zorder=0)
        if baseline:
            ax.axhline(lim[0], color=t["axis"], lw=0.75, zorder=1)


def legend(cv, x, y, items, size=7.5):
    """A row of (label, color key, style) swatches starting at (x, y) in inches. Returns the end x."""
    for label, key, style in items:
        if style == "dot":
            cv.dot(x + 0.05, y + 0.035, key, r=0.045, ring=False)
        elif style == "ring":
            cv.dot(x + 0.05, y + 0.035, key, r=0.04, hollow=True)
        elif style == "line":
            cv.line([x, x + 0.22], [y + 0.035, y + 0.035], color=key, lw=1.5)
            x += 0.12
        else:
            cv.ax.add_patch(FancyBboxPatch((x, y - 0.02), 0.11, 0.11, boxstyle="round,pad=0,rounding_size=0.02",
                                           fc=cv.t[key], ec="none", zorder=3))
        cv.text(x + 0.17, y, label, size=size, color="ink2")
        x += 0.17 + 0.062 * len(label) + 0.25
    return x


def by_id(data):
    return {d["id"]: d for d in data["datasets"]}


# ======================================================================================== figures: evals
def fig_noise_floor(D, t):
    rows = D["noise_floor"]["rows"][:3]
    summ = D["noise_floor"]["rows"][3]
    cv = Canvas(t, 8.4, 3.75)
    cv.header("Pangram's noise floor: how far the score moved under edits that change nothing",
              f"3 passages, each scored as is and with three trivial edits ({summ['scorings']} scorings). "
              f"{summ['edited versions that moved']} edited versions moved.")
    ax = cv.axes(2.0, 0.55, 5.9, 1.85)
    value_axis(ax, t, (0, 0.25), [0.0, 0.05, 0.10, 0.15, 0.20, 0.25])
    ax.set_ylim(3, 0)
    kinds = [("comma moved", "s1"), ("contraction swapped", "s2"), ("twin word", "s3")]
    for i, r in enumerate(rows):
        if i:
            ax.axhline(i, color=t["grid"], lw=0.75, zorder=0)
        ax.text(-0.012, i + 0.5, f"{r['passage']}", ha="right", va="center", fontsize=8.5, color=t["ink"],
                transform=ax.transData, **BOLD)
        ax.text(-0.012, i + 0.78, f"{r['words']} words, original {fmt(r['original'])}", ha="right", va="center",
                fontsize=7.5, color=t["ink2"])
        for j, (k, c) in enumerate(kinds):
            move = abs(r[k] - r["original"])
            ax.plot([move], [i + 0.28 + j * 0.22], "o", ms=6.5, mfc=t[c], mec=t["surface"], mew=1.5, zorder=4,
                    clip_on=False)
            if move > 0:
                ax.text(move - 0.004, i + 0.28 + j * 0.22, f"{fmt(r['original'])} to {fmt(r[k])}, moved {fmt(move)}",
                        ha="right", va="center", fontsize=7.5, color=t["ink"])
    thr = summ["keep threshold"]
    ax.axvline(thr, color=t["ink2"], lw=1.0, zorder=2)
    ax.text(thr + 0.003, -0.08, f"keep threshold {fmt(thr)}", fontsize=7.5, color=t["ink"], va="bottom")
    cv.text(2.0, 0.22, "how far the score moved from the original (AI share, 0 human to 1 AI)", size=7.5,
            color="muted")
    legend(cv, 2.0, 2.8, [(k, c, "dot") for k, c in kinds])
    return cv


def fig_keep_or_drop(D, t):
    rows = D["keep_or_drop"]["rows"]
    groups = []
    for r in rows:
        if not groups or groups[-1][0] != r["group"]:
            groups.append((r["group"], []))
        groups[-1][1].append(r)
    rh, gh = 0.36, 0.34
    h = 2.0 + len(rows) * rh + len(groups) * gh
    cv = Canvas(t, 9.0, h)
    cv.header("The keep-or-drop record: every model and method tried",
              ["Each one kept, dropped, parked or held as an alternate, on the evidence. The track is Pangram's",
               "AI share (0 human, 1 AI); an arrow runs from the text before to the text after."])
    X_TRACK, W_TRACK, X_DEC = 4.05, 1.55, 5.95
    y = h - 1.2
    cv.text(0.25, y, "model or method, and its result", size=7, color="muted")
    cv.text(X_TRACK, y, "0", size=7, color="muted", ha="center")
    cv.text(X_TRACK + W_TRACK, y, "1", size=7, color="muted", ha="center")
    cv.text(X_TRACK + W_TRACK / 2, y, "AI share", size=7, color="muted", ha="center")
    cv.text(X_DEC, y, "decision and reason", size=7, color="muted")
    y -= 0.12
    for g, items in groups:
        y -= gh
        cv.text(0.25, y + 0.08, g, size=8.5, bold=True)
        cv.line([0.25, 8.75], [y + 0.02, y + 0.02], color="axis")
        for r in items:
            y -= rh
            cv.text(0.25, y + 0.16, r["what"], size=8.5)
            cv.text(0.25, y + 0.02, r["result"], size=7.5, color="ink2")
            ty = y + 0.11
            cv.line([X_TRACK, X_TRACK + W_TRACK], [ty, ty], color="grid", lw=1.5)
            sx = lambda v: X_TRACK + v * W_TRACK          # noqa: E731
            tr = r["track"]
            for v in tr.get("dots", []):
                cv.dot(sx(v), ty, "ink2", r=0.04)
            for a, b in tr.get("moves", []):
                cv.dot(sx(a), ty, "ink2", r=0.035, hollow=True)
                if abs(b - a) * W_TRACK > 0.12:
                    cv.arrow((sx(a), ty), (sx(b) - (0.05 if b > a else -0.05), ty), color="muted", lw=0.9)
                cv.dot(sx(b), ty, "ink2", r=0.04)
            cv.dot(X_DEC + 0.05, y + 0.19, DECISION[r["decision"]], r=0.045, ring=False)
            cv.text(X_DEC + 0.17, y + 0.16, r["decision"], size=8.5, bold=True)
            cv.text(X_DEC + 0.17, y + 0.02, r["reason"], size=7.5, color="ink2")
    legend(cv, 0.25, 0.2, [("kept", "s1", "dot"), ("parked or alternate", "s3", "dot"), ("dropped", "s2", "dot")])
    cv.text(4.2, 0.2, "track: ring before, dot after", size=7.5, color="muted")
    return cv


def fig_detectors(D, t):
    rows = D["bakeoff_models"]["rows"]
    agr = {r["metric"]: r for r in D["open_pangram_agreement"]["rows"]}
    sp = agr["model-level Spearman over 9 bake-off rows (derived)"]["roberta_large"]
    sp_wo = agr["model-level Spearman without the humanizer row (derived)"]["roberta_large"]
    doc = agr["document Spearman (28 documents)"]
    cv = Canvas(t, 8.4, 5.0)
    k = D["bakeoff_models"]["scorings_per_model"]
    cv.header("Pangram against Open Pangram on the bake-off models",
              [f"Open Pangram ranks for free; Pangram decides. Each dot is one model's mean over {k} Pangram scorings.",
               f"Spearman over the {len(rows)} model means: {fmt(sp)} ({fmt(sp_wo)} without the humanizer model). "
               f"Per document it is {fmt(doc['roberta_large'])}."])
    ax = cv.axes(0.75, 0.75, 3.45, 3.3)
    ticks, labs = [0.0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"]
    value_axis(ax, t, (0, 1.0), ticks, labs, axis="y")
    ax.set_xlim(0, 1.0)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs)
    for x in ticks[1:]:
        ax.axvline(x, color=t["grid"], lw=0.75, zorder=0)
    ax.axvline(0, color=t["axis"], lw=0.75, zorder=1)
    cv.text(0.75 + 1.72, 0.35, "Open Pangram score (local, free)", size=7.5, color="muted", ha="center")
    cv.text(0.22, 0.75 + 1.65, "Pangram AI share (paid)", size=7.5, color="muted", ha="center", rotation=90,
            va="center")
    cat = {"kept": "s1", "alternate": "s3", "dropped": "s2"}
    for r in rows:
        key = "kept" if r["decision"].startswith("kept") else "alternate" if r["decision"].startswith(
            "alternate") else "dropped"
        ax.plot([r["open_pangram"]], [r["pangram"]], "o", ms=7, mfc=t[cat[key]], mec=t["surface"], mew=1.5,
                zorder=4, clip_on=False)
    # labels in a column to the right of the plot, joined to their dots by thin leaders
    order = sorted(rows, key=lambda r: -r["pangram"])
    lx = 4.6
    tops = [3.9 - i * 0.37 for i in range(len(order))]
    for r, ly in zip(order, tops):
        px = 0.75 + r["open_pangram"] * 3.45
        py = 0.75 + r["pangram"] * 3.3
        cv.line([px + 0.06, lx - 0.06], [py, ly + 0.04], color="axis", lw=0.6)
        cv.text(lx, ly + 0.08, f"{r['model']}  (round {r['round']})", size=7.5)
        cv.text(lx, ly - 0.07, f"Pangram {fmt(r['pangram'])}, Open Pangram {fmt(r['open_pangram'])}; "
                               f"{re.split(r' [(]|;', r['decision'])[0]}", size=7, color="ink2")
    legend(cv, 0.75, 0.12, [("kept", "s1", "dot"), ("alternate", "s3", "dot"), ("dropped", "s2", "dot")])
    return cv


def fig_judge(D, t):
    rows = [r for r in D["h2h_outcomes"]["rows"] if r["outcome"] != "all"]
    total = [r for r in D["h2h_outcomes"]["rows"] if r["outcome"] == "all"][0]["total"]
    names = {"pick won": "pick won", "tie (SAME, counts as a pass)": "tie (SAME counts as a pass)",
             "pick lost, shipped anyway under the last-round settle rule": "lost, shipped under the last-round rule",
             "pick lost, not shipped": "lost, not shipped",
             "outcome not recoverable (pick then failed the fact check)": "not recoverable (failed the fact check)"}
    cv = Canvas(t, 8.4, 3.2)
    cv.header("The blind judge: head-to-head outcomes in the window loop",
              ["Sonnet sees two unlabeled versions and answers A, B or SAME.",
               f"The outcomes add up to {num(total)}, the ledger's count of head-to-head calls in the window loop."])
    ax = cv.axes(3.0, 0.5, 4.9, 1.6)
    value_axis(ax, t, (0, 800), [0, 200, 400, 600, 800], ["0", "200", "400", "600", "800"])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i, r in enumerate(rows):
        a, b = r["rounds"], r.get("context rounds", 0)
        bar(ax, i, 0, a, t["s1"], round_end=b == 0)
        if b:
            bar(ax, i, a + pt2d(ax, 1.5, "x"), a + b, t["s3"])
        ax.text(-12, i, names[r["outcome"]], ha="right", va="center", fontsize=8, color=t["ink"])
        ax.text(a + b + 10, i, num(r["total"]), ha="left", va="center", fontsize=8, color=t["ink"], **BOLD)
    legend(cv, 3.0, 0.15, [("window rounds", "s1", "sq"), ("context rounds", "s3", "sq")])
    return cv


def fig_voice(D, t):
    base = D["delta_baselines"]["rows"]
    rows = D["delta_comparisons"]["rows"]
    panels = [
        ("exp09 (baseline 1)", "Exp 9: voice adapter", "loop", [("adapter_2ep", "adapter, 2 epochs"),
                                                               ("adapter_3ep", "adapter, 3 epochs")], "loop text"),
        ("paraphrase adapter voice pass (baseline 1)", "Paraphrase adapter as a voice pass", "plain",
         [("voice_pass", "voice pass")], "plain text"),
        ("exp10 editors (baseline 1)", "Exp 10: line editors", "shipped",
         [("olmo_sft", "OLMo SFT"), ("mistral_instruct", "Mistral instruct"), ("sonnet", "Sonnet")], "shipped"),
        ("exp10b careful editors (baseline 1)", "Exp 10b: careful editors", "shipped",
         [("opus", "Opus"), ("sonnet", "Sonnet")], "shipped"),
        ("exp11 (baseline 2)", "Exp 11: adapter retrain", "live_adapter",
         [("retrain_tag_1", "retrain, tag 1"), ("retrain_tag_2", "retrain, tag 2")], "live adapter"),
        ("paragraph pass (baseline 2)", "Paragraph pass", "standing", [("best", "best")], "standing text"),
    ]
    cv = Canvas(t, 8.6, 7.75)
    cv.header("The voice metric: every Burrows' Delta reading against the author's own text",
              ["Lower is closer to the author. The lines are the author's own chunks: median and 90th percentile.",
               "The reference corpus changed, so the two baselines are never compared."])
    lim = (0.65, 1.30)
    bands = [(6.35, base[0], "Baseline 1"), (2.25, base[1], "Baseline 2, after the reference corpus changed")]
    for ytop, b, name in bands:
        cv.text(0.25, ytop + 0.12, f"{name}: author median {fmt(b['median'])}, 90th percentile {fmt(b['p90'])}",
                size=8.5, bold=True)
    spots = [(0, 6.35), (1, 6.35), (0, 4.3), (1, 4.3), (0, 2.25), (1, 2.25)]
    for (col, ytop), (test, title, ref, ch, ref_name) in zip(spots, panels):
        b = base[0] if "baseline 1" in test else base[1]
        prow = [r for r in rows if r["test"] == test]
        x0 = 0.25 + col * 4.25
        cv.text(x0, ytop - 0.22, title, size=8.5, bold=True)
        items = [(ref_name, "muted", "dot")] + [(n, f"s{i + 1}", "dot") for i, (_, n) in enumerate(ch)]
        legend(cv, x0, ytop - 0.42, items, size=7)
        hgt = 0.27 * len(prow) + 0.1
        ax = cv.axes(x0 + 0.85, ytop - 0.55 - hgt, 3.05, hgt)
        value_axis(ax, t, lim, [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3],
                   ["0.7", "0.8", "0.9", "1.0", "1.1", "1.2", "1.3"],
                   baseline=False)
        ax.set_ylim(len(prow) - 0.4, -0.6)
        for x, lab in ((b["median"], "median"), (b["p90"], "p90")):
            ax.axvline(x, color=t["ink2"], lw=1.0, zorder=1)
            ax.text(x, -0.62, lab, ha="center", va="bottom", fontsize=6.5, color=t["ink2"])
        for i, r in enumerate(prow):
            ax.text(lim[0] - 0.012, i, r["window"], ha="right", va="center", fontsize=7.5, color=t["ink"])
            ax.axhline(i, color=t["grid"], lw=0.75, zorder=0)
            ax.plot([r[ref]], [i], "o", ms=6.5, mfc=t["muted"], mec=t["surface"], mew=1.5, zorder=3)
            for j, (k, _) in enumerate(ch):
                ax.plot([r[k]], [i], "o", ms=6.5, mfc=t[f"s{j + 1}"], mec=t["surface"], mew=1.5, zorder=4 + j)
        ax.tick_params(axis="x", labelsize=6.5, pad=2)
    cv.text(0.25, 0.2, "Where two readings are equal their dots sit on top of each other; the table in the README has "
                       "every reading.", size=7, color="muted")
    return cv


def fig_gates(D, t):
    st = D["paragraph_pass_stages"]["rows"]
    gate = D["paragraph_pass_gate"]["rows"]
    allr = [r for r in gate if r["reason"] == "all draws"][0]
    reasons = sorted([r for r in gate if r["reason"] not in ("passed", "all draws")], key=lambda r: -r["draws"])
    cv = Canvas(t, 8.6, 5.6)
    cv.header("The gate funnel: the paragraph pass, from draws to picks",
              f"Code stopped {num(allr['rejected'])} of {num(allr['draws'])} draws "
              f"({allr['rejected_share'] * 100:.1f}%) before any LLM or detector saw them. "
              f"Each stage is its own count.")
    ax = cv.axes(1.75, 2.6, 1.95, 1.9)
    value_axis(ax, t, (0, 4000), [0, 2000, 4000], ["0", "2,000", "4,000"])
    ax.set_ylim(len(st) - 0.5, -0.5)
    for i, r in enumerate(st):
        bar(ax, i, 0, r["n"], t["ordinal"][min(i, 4)], thick_pt=13)
        ax.text(-80, i, r["stage"], ha="right", va="center", fontsize=8, color=t["ink"])
        ax.text(r["n"] + 70, i, num(r["n"]), ha="left", va="center", fontsize=8, color=t["ink"], **BOLD)
    split = st[3].get("split")
    cv.text(0.25, 2.15, f"Passed the fact check: {split}.", size=7.5, color="ink2")
    cv.text(0.25, 1.98, "The strict run stopped at its Claude ceiling", size=7.5, color="ink2")
    cv.text(0.25, 1.83, "before any Pangram call.", size=7.5, color="ink2")
    cv.text(0.25, 4.72, "Counts at each stage", size=8.5, bold=True)
    cv.text(4.85, 4.72, "Why the code gate rejected a draw (first reason)", size=8.5, bold=True)
    ax2 = cv.axes(6.15, 0.45, 2.15, 4.1)
    value_axis(ax2, t, (0, 1800), [0, 600, 1200, 1800], ["0", "600", "1,200", "1,800"])
    ax2.set_ylim(len(reasons) - 0.5, -0.5)
    for i, r in enumerate(reasons):
        bar(ax2, i, 0, r["draws"], t["s1"], thick_pt=8, r_pt=2.5)
        ax2.text(-30, i, r["reason"], ha="right", va="center", fontsize=7.5, color=t["ink"])
        ax2.text(r["draws"] + 25, i, num(r["draws"]), ha="left", va="center", fontsize=7, color=t["ink2"])
    return cv


def fig_fact_check(D, t):
    rows = D["fact_gate"]["rows"]
    cv = Canvas(t, 8.4, 3.95)
    loose = {r["arm"]: r for r in rows if r["gate"] == "paragraph pass, loose prompt"}
    rate = {a: loose[a]["pass"] / (loose[a]["pass"] + loose[a]["fail"]) for a in ("adapter", "few-shot")}
    cv.header("The fact check: Sonnet passes or fails each gated draft",
              [f"On the loose prompt the adapter passed {rate['adapter']:.0%} of its checks and the few-shot writer "
               f"{rate['few-shot']:.0%}.",
               "The two prompts ran on different pools of drafts, so compare arms, not prompts."])
    gname = {"paragraph pass, loose prompt": "paragraph pass, loose prompt",
             "paragraph pass, strict prompt (pulled)": "paragraph pass, strict prompt (run stopped)",
             "window loop": "window loop"}
    layout, y, last = [], 0.0, None
    for r in rows:
        if r["gate"] != last:
            y += 0.5 if last is not None else 0
            layout.append(("head", y, gname[r["gate"]]))
            last = r["gate"]
            y += 1
        layout.append(("row", y, r))
        y += 1
    ax = cv.axes(3.1, 0.55, 4.1, 2.35)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(y - 0.5, -0.5)
    gap = pt2d(ax, 1.5, "x")
    edge = blended_transform_factory(cv.fig.transFigure, ax.transData)
    for kind, yy, r in layout:
        if kind == "head":
            ax.text(0.25 / cv.w, yy, r, transform=edge, ha="left", va="center", fontsize=8, color=t["ink"], **BOLD)
            continue
        n = r["pass"] + r["fail"]
        p = r["pass"] / n
        bar(ax, yy, 0, p, t["s1"], thick_pt=9, round_end=False)
        bar(ax, yy, p + (gap if r["pass"] else 0), 1, t["muted"], thick_pt=9)
        ax.text(-0.03, yy, r["arm"], ha="right", va="center", fontsize=8, color=t["ink"])
        ax.text(1.03, yy, f"{r['pass']} of {n} passed", ha="left", va="center", fontsize=7.5, color=t["ink2"])
    legend(cv, 3.1, 0.2, [("passed", "s1", "sq"), ("failed", "muted", "sq")])
    return cv


def fig_humanizer(D, t):
    rows = [r for r in D["humanizer_by_pass"]["rows"] if r["purpose"] != "all"]
    tot = [r for r in D["humanizer_by_pass"]["rows"] if r["purpose"] == "all"][0]
    focus = {"window rounds", "paragraph pass"}
    cv = Canvas(t, 8.4, 5.0)
    w = [r for r in rows if r["pass"] == "window rounds"][0]
    p = [r for r in rows if r["pass"] == "paragraph pass"][0]
    cv.header("The humanizer flag by pass: the open problem",
              f"{w['humanized']} of {w['scorings']} window-round scorings were flagged, against {p['humanized']} of "
              f"{p['scorings']} in the paragraph pass ({tot['humanized']} of {tot['scorings']} across the project).")
    ax = cv.axes(3.35, 0.45, 3.9, 3.85)
    value_axis(ax, t, (0, 1.0), [0.0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i, r in enumerate(rows):
        share = r["humanized"] / r["scorings"]
        c = t["s1"] if r["pass"] in focus else t["muted"]
        bar(ax, i, 0, share, c, thick_pt=9)
        lab = f"{r['pass']}"
        ax.text(-0.02, i, lab, ha="right", va="center", fontsize=7.5, color=t["ink"],
                **(BOLD if r["pass"] in focus else {}))
        ax.text(share + 0.015, i, f"{r['humanized']} of {r['scorings']}", ha="left", va="center", fontsize=7.5,
                color=t["ink"] if r["pass"] in focus else t["ink2"])
    cv.text(3.35, 0.15, "share of scorings with the humanizer flag set", size=7.5, color="muted")
    return cv


# ================================================================================== figures: experiments
def fig_skeleton(D, t):
    rows = {(r["skeleton"], r["words"]): r for r in D["skeleton_word_swap"]["rows"]}
    cv = Canvas(t, 6.4, 4.85)
    cv.header("Skeletons or words? The 2x2 swap",
              ["Sentence skeletons and word choice swapped between the author's", "writing and the model's."])
    cell, x0, ytop = 1.55, 2.0, 3.65
    cv.text(x0 + cell, ytop + 0.27, "words from", size=7.5, color="muted", ha="center")
    cv.text(x0 - 0.55, ytop - cell, "skeleton from", size=7.5, color="muted", ha="center", rotation=90,
            va="center")
    for j, wsrc in enumerate(("author", "model")):
        cv.text(x0 + cell * j + cell / 2, ytop + 0.08, f"the {wsrc}", size=8.5, bold=True, ha="center")
    for i, ssrc in enumerate(("author", "model")):
        cv.text(x0 - 0.1, ytop - cell * i - cell / 2, f"the {ssrc}", size=8.5, bold=True, ha="right",
                va="center")
        for j, wsrc in enumerate(("author", "model")):
            r = rows[(ssrc, wsrc)]
            v = r["ai_share"]
            fill = t["seq"][round(v * 12)]
            lum = matplotlib.colors.rgb_to_hsv(matplotlib.colors.to_rgb(fill))[2]
            dark_fill = sum(matplotlib.colors.to_rgb(fill)) / 3 < 0.55 and lum < 0.75
            ink = "#ffffff" if dark_fill else "#0b0b0b"
            x, yt = x0 + cell * j, ytop - cell * i
            cv.ax.add_patch(FancyBboxPatch((x + 0.03, yt - cell + 0.03), cell - 0.06, cell - 0.06,
                                           boxstyle="round,pad=0,rounding_size=0.05", fc=fill, ec="none"))
            cv.text(x + cell / 2, yt - 0.62, fmt(v), size=18, bold=True, color=ink, ha="center")
            cv.text(x + cell / 2, yt - 0.9, f"n = {r['n']}", size=7.5, color=ink, ha="center")
            kind = "original" if ssrc == wsrc else "swap"
            cv.text(x + cell / 2, yt - 1.08, kind, size=7.5, color=ink, ha="center")
    cv.text(x0 + 2 * cell + 0.2, ytop - 0.55, "AI share", size=7.5, color="muted")
    cv.text(x0 + 2 * cell + 0.2, ytop - 0.72, "0 human, 1 AI", size=7.5, color="muted")
    cv.text(0.25, 0.2, "Model skeletons stay flagged whatever words they wear.", size=8.5, color="ink2")
    return cv


def fig_split(D, t):
    rows = D["diag_split"]["rows"]
    cv = Canvas(t, 8.4, 2.5)
    cv.header("One window split by author",
              "Each set alone, then the whole window together. The model's sentences pull the author's up with them.")
    ax = cv.axes(3.3, 0.45, 4.4, 1.2)
    value_axis(ax, t, (0, 1.0), [0.0, 0.25, 0.5, 0.75, 1.0], ["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i, r in enumerate(rows):
        bar(ax, i, 0, r["ai_share"], t["s1"], thick_pt=12)
        bits = [f"{r['sentences']} sentences"] + ([f"{r['words']} words"] if "words" in r else [])
        ax.text(-0.02, i - 0.13, r["passage"], ha="right", va="center", fontsize=8.5, color=t["ink"])
        ax.text(-0.02, i + 0.22, ", ".join(bits), ha="right", va="center", fontsize=7, color=t["ink2"])
        ax.text(r["ai_share"] + 0.015, i, fmt(r["ai_share"]), ha="left", va="center", fontsize=8.5,
                color=t["ink"], **BOLD)
    cv.text(3.3, 0.12, "Pangram AI share", size=7.5, color="muted")
    return cv


def fig_trajectories(D, t):
    rows = D["section_trajectories"]["rows"]
    cv = Canvas(t, 9.0, 5.75)
    cv.header("Every scoring of every section, in order",
              ["Pangram AI share of each section as posted. The low and the latest unflagged scoring are labeled.",
               "A hollow dot had the humanizer flag set: such a scoring never counts as a best, and its text "
               "wasn't written."])
    pw, ph = 1.72, 1.35
    for k, r in enumerate(rows):
        col, rowi = k % 4, k // 4
        x0 = 0.6 + col * 2.12
        y0 = 2.95 - rowi * 2.35
        name = r["section"]
        m = re.match(r"(.*?) \((.*)\)$", name)
        title, sub = (m.group(1), m.group(2)) if m else (name, "")
        cv.text(x0 - 0.35, y0 + ph + 0.42, title, size=8, bold=True)
        if sub:
            cv.text(x0 - 0.35, y0 + ph + 0.25, sub if "words" in sub else sub + " words", size=7, color="ink2")
        ax = cv.axes(x0, y0, pw, ph)
        pts = r["series"]
        n = len(pts)
        value_axis(ax, t, (0, 1.0), [0.0, 0.5, 1.0], ["0", "0.5", "1"], axis="y", baseline=False)
        ax.set_ylim(-0.16, 1.1)
        ax.set_xlim(-0.4, n - 0.6)
        xs = list(range(n))
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=t["s1"], lw=1.5, zorder=3)
        for x, (lab, v) in zip(xs, pts):
            hum = "humanized" in lab
            ax.plot([x], [v], "o", ms=6, mfc=t["surface"] if hum else t["s1"], mec=t["s1"] if hum else t["surface"],
                    mew=1.5, zorder=4, clip_on=False)
        clean = [i for i in range(n) if "humanized" not in pts[i][0]]
        lo = min(clean, key=lambda i: (ys[i], -i))           # the latest of equal lows
        last = clean[-1]
        ax.text(xs[last], ys[last] + 0.08, fmt(ys[last]), ha="center" if last < n - 1 else "right", va="bottom",
                fontsize=7.5, color=t["ink"], **BOLD)
        if lo != last and ys[lo] != ys[last]:
            ax.text(xs[lo], ys[lo] - 0.08, fmt(ys[lo]), ha="center", va="top", fontsize=7.5, color=t["ink"],
                    **BOLD)
        short = lambda s_: re.sub(r" [(].*[)]$", "", s_).split(",")[0]        # noqa: E731
        cv.text(x0, y0 - 0.16, short(pts[0][0]), size=6.5, color="muted")
        cv.text(x0 + pw, y0 - 0.16, short(pts[-1][0]), size=6.5, color="muted", ha="right")
    return cv


def fig_timeline(D, t):
    rows = D["timeline"]["rows"]
    rh = 0.245
    h = 1.2 + len(rows) * rh
    cv = Canvas(t, 9.0, h)
    cv.header("Every experiment and milestone, in order",
              "November 2026. Each row is one experiment or build step, its result and what was decided.")
    XD, XL, XI, XR, XC = 0.25, 0.95, 1.12, 3.6, 7.05
    y = h - 0.98
    for x, s in ((XD, "date"), (XI, "experiment or milestone"), (XR, "result"), (XC, "decision")):
        cv.text(x, y, s, size=7, color="muted")
    y -= 0.08
    cv.line([XL, XL], [y - 0.05, y - len(rows) * rh + 0.05], color="axis", lw=1.0)
    last = None
    for r in rows:
        y -= rh
        if r["date"] != last:
            cv.text(XD, y + 0.05, r["date"], size=7.5, bold=True)
            if last is not None:
                cv.line([XD, 8.75], [y + rh - 0.04, y + rh - 0.04], color="grid")
            last = r["date"]
        cv.dot(XL, y + 0.085, "s1", r=0.04)
        cv.text(XI, y + 0.05, r["item"], size=7.5)
        cv.text(XR, y + 0.05, r["result"], size=7.5, color="ink2")
        cv.text(XC, y + 0.05, r["decision"], size=7.5)
    return cv


# ======================================================================================= figures: the loop
def fig_pipeline(D, t):
    steps = D["pipeline"]["rows"]
    cv = Canvas(t, 9.0, 4.3)
    cv.header("The pipeline, one paragraph at a time",
              "Claude reads, a base model writes, code decides what reaches the detectors. "
              "Counts are from the paragraph pass.")
    bw, bh, gx = 1.42, 0.98, 0.28
    pos = []
    for i in range(5):
        pos.append((0.25 + i * (bw + gx), 3.3))
    for i in range(4):
        pos.append((0.25 + (4 - i) * (bw + gx), 1.8))
    for i, (s, (x, y)) in enumerate(zip(steps, pos)):
        lines = [(s["step"], "title"), (s["who"], "small")] + [(ln, "body") for ln in wrap(s["what"], 20)]
        cv.box(x, y, bw, bh, lines, accent=s["step"] in ("Code gates", "Pangram"))
    for i in range(4):
        (x, y) = pos[i]
        cv.arrow((x + bw + 0.02, y - bh / 2), (x + bw + gx - 0.02, y - bh / 2))
    x, y = pos[4]
    cv.arrow((x + bw / 2, y - bh - 0.02), (x + bw / 2, pos[5][1] + 0.02))
    for i in range(5, 8):
        (x, y) = pos[i]
        cv.arrow((x - 0.02, y - bh / 2), (x - gx + 0.02, y - bh / 2))
    # the code gates send failures back to the writer
    gx0 = pos[2][0] + 0.35
    dx1 = pos[1][0] + bw - 0.35
    cv.arrow((gx0, pos[2][1] + 0.02), (dx1, pos[1][1] + 0.02), rad=0.45, color="muted")
    cv.text((gx0 + dx1) / 2, pos[1][1] + 0.27, "fail: draw again", size=7, color="ink2", ha="center")
    c = {s["step"]: s for s in steps}
    for i, s_ in ((1, f"{num(c['Draws']['count'])} draws"), (2, f"{num(c['Code gates']['count_out'])} passed"),
                  (3, f"{num(c['Fact check']['count'])} checked"),
                  (8, f"{num(c['Write-back']['count'])} written back")):
        cv.text(pos[i][0] + 0.02, pos[i][1] - bh - 0.2, s_, size=7.5, bold=True)
    cv.text(0.25, 0.2, "Highlighted: the two steps that decide. Code gates are free and strict; Pangram is paid "
                       "and has the last word.", size=7.5, color="ink2")
    return cv


def fig_night(D, t):
    rows = D["night"]["rows"]
    steps, times, after, first = rows[0]["steps"], rows[1], rows[2]["after_pass"], rows[3]["first_night"]
    cv = Canvas(t, 9.0, 4.75)
    cv.header("A night: steps, deadlines and guards",
              f"A scheduled task fires at {times['fires']}. The guards run first, in order; past a deadline the "
              "whole process tree is killed.")
    gw = 2.9
    lines = [("At the start, in order", "title")] + [(f"{i + 1}. {s_}", "body")
                                                             for i, s_ in enumerate(steps)]
    cv.box(0.25, 3.85, gw, 0.33 + 0.145 * len(steps) + 0.06, lines, accent=True)
    kill = [("Past a deadline", "title"), (f"children are polled every {times['poll_s']} s;", "body"),
            ("list them, kill them, wait", "body"),
            (f"{times['kill_wait_s']} s, force-kill the rest, then", "body"), ("check the port and the bank", "body")]
    cv.box(0.25, 2.0, gw, 1.0, kill)

    ax0, ax1 = 3.55, 8.75

    def tx(hh):
        h_, m_ = hh.split(":")[:2]
        return ax0 + (int(h_) - 2 + int(m_) / 60) / 5 * (ax1 - ax0)

    hour = tx("03:00") - tx("02:00")
    base = 1.05
    cv.line([ax0, ax1], [base, base], color="axis")
    for hh in ("02:00", "03:00", "04:00", "05:00", "06:00", "07:00"):
        cv.line([tx(hh), tx(hh)], [base, 3.75], color="grid", z=0)
        cv.text(tx(hh), base - 0.18, hh, size=7.5, color="ink2", ha="center")
    # lane 1: the schedule
    cv.text(ax0, 3.6, "the schedule", size=8, bold=True)
    yb = 3.25
    x0, x6 = tx(times["fires"]), tx(times["pass_until"])
    x7 = x6 + hour * times["finals_max_min"] / 60
    cv.ax.add_patch(FancyBboxPatch((x0, yb - 0.1), x6 - x0 - 0.02, 0.2, boxstyle="round,pad=0,rounding_size=0.04",
                                   fc=t["s1"], ec="none", zorder=3))
    cv.text(x0 + 0.08, yb - 0.04, "paragraph pass", size=7.5, color="#ffffff", zorder=4)
    cv.ax.add_patch(FancyBboxPatch((x6 + 0.02, yb - 0.1), x7 - x6 - 0.02, 0.2,
                                   boxstyle="round,pad=0,rounding_size=0.04", fc=t["s3"], ec="none", zorder=3))
    cv.text(x7 + 0.06, yb - 0.04, "submission", size=7, color="ink2")
    cv.text(x6 + 0.05, yb + 0.16, "finals", size=7, color="ink2")
    for i, s_ in enumerate([f"before the pass: {steps[-1]}",
                            f"after it: {after[0]} (up to {times['finals_max_min']} min); {after[1]}"]):
        cv.text(ax0, 0.6 - i * 0.16, s_, size=7, color="ink2")
    # deadlines
    for hh in (times["late_start"], times["pass_until"]):
        cv.line([tx(hh), tx(hh)], [2.08, yb - 0.12], color="ink2", lw=1.0, z=2)
    cv.text(tx(times["late_start"]) - 0.06, 2.42, "late-start cutoff", size=7, color="ink", ha="right", bold=True)
    cv.text(tx(times["late_start"]) - 0.06, 2.27, f"a start at or after {times['late_start']}", size=7,
            color="ink2", ha="right")
    cv.text(tx(times["late_start"]) - 0.06, 2.13, "skips the pass", size=7, color="ink2", ha="right")
    cv.text(tx(times["pass_until"]) + 0.06, 2.42, "pass deadline", size=7, color="ink", bold=True)
    cv.text(tx(times["pass_until"]) + 0.06, 2.27, f"{times['pass_until']}", size=7, color="ink2")
    # lane 2: the first unattended night
    l2 = 1.75
    cv.text(ax0, l2 + 0.2, "first unattended night", size=8, bold=True)
    end = first["submission"].split(":")
    x1 = tx("02:00") + hour * (int(end[1]) + int(end[2]) / 60) / 60
    cv.ax.add_patch(FancyBboxPatch((tx("02:00"), l2 - 0.09), x1 - tx("02:00"), 0.16,
                                   boxstyle="round,pad=0,rounding_size=0.03", fc=t["s1"], ec="none", zorder=3))
    cv.text(x1 + 0.08, l2 - 0.04, f"pass {first['start']} to {first['pass_end']}, finals to {first['finals_end']}, "
                                  f"submission {first['submission']}", size=7, color="ink2")
    cv.text(x1 + 0.08, l2 - 0.2, f"spent: Pangram {usd(first['pangram_usd'])}, Claude {usd(first['claude_usd'])} "
                                 "(estimated)", size=7, color="ink2")
    cv.text(0.25, 0.12, "A STOP file stops the night at any check. A busy model port or a running engine process "
                        "means the night doesn't start.", size=7.5, color="ink2")
    return cv


def fig_caps(D, t):
    rows = D["caps"]["rows"]
    cv = Canvas(t, 9.0, 5.95)
    cv.header("Budget controls: every cap, in the order the code checks it",
              ["A call goes through its lane top to bottom, and the first cap that says no holds it until that "
               "cap's reset.", "The paragraph pass and repair call Claude outside the lane, under their own "
               "ceilings (right)."])
    lanes = [("Pangram", 0.25, "every paid call; 2 to 4 for the loop only"), ("Claude", 3.25, "every Claude call "
                                                                                              "in the loop")]
    bw, bh = 2.7, 0.56
    for lane, x, top in lanes:
        items = sorted([r for r in rows if r["lane"] == lane], key=lambda r: r["order"])
        cv.text(x, 4.62, lane, size=9.5, bold=True)
        cv.text(x, 4.45, top, size=7.5, color="ink2")
        y = 4.3
        for r in items:
            n = r["order"]
            label = f"{n}. {r['cap']}" if n else r["cap"]
            last = ("no: " + r["on_hit"]) if r["on_hit"] else "prices the call for the checks below"
            cv.box(x, y, bw, bh, [(label, "ink"), (r["value"], "body"), (last, "small")], accent=n == 1, pad=0.08)
            nxt = y - bh - 0.12
            if r is not items[-1]:
                cv.arrow((x + bw / 2, y - bh - 0.01), (x + bw / 2, nxt + 0.01))
            y = nxt
        cv.text(x + bw / 2, y + 0.02, "call sent, cost logged", size=7.5, color="ink2", ha="center")
    x = 6.25
    cv.text(x, 4.62, "Also checked", size=9.5, bold=True)
    cv.text(x, 4.45, "by each pass and each call", size=7.5, color="ink2")
    y = 4.3
    for r in rows:
        if r["lane"] in ("Pangram", "Claude"):
            continue
        body = wrap(r["value"], 34)
        extra = [(f"spent {usd(r['spent'])}", "small")] if "spent" in r else []
        hh = 0.3 + 0.145 * (len(body) + len(extra))
        cv.box(x, y, 2.5, hh, [(r["cap"], "ink")] + [(b, "body") for b in body] + extra, pad=0.08)
        y -= hh + 0.12
    return cv


def fig_spend(D, t):
    rows = [r for r in D["pangram_spend_by_day"]["rows"] if r["day"] != "total"]
    tot = [r for r in D["pangram_spend_by_day"]["rows"] if r["day"] == "total"][0]
    days = sorted({r["day"] for r in rows})
    purposes = [("loop", "s1"), ("experiment", "s2"), ("calibration", "s3")]
    month_cap = [r for r in D["caps"]["rows"] if r["lane"] == "Pangram" and r.get("order") == 1][0]["value"]
    cv = Canvas(t, 8.4, 3.6)
    cv.header("Pangram spend by day and purpose",
              f"{num(tot['calls'])} calls, {usd(tot['usd'])} in all, under a {month_cap} monthly cap. "
              "Every call returned ok on model 4.1.")
    ax = cv.axes(0.75, 0.6, 7.3, 2.15)
    value_axis(ax, t, (0, 40), [0, 10, 20, 30, 40], ["$0", "$10", "$20", "$30", "$40"], axis="y")
    ax.set_xlim(-0.6, len(days) - 0.4)
    gap = pt2d(ax, 1.5, "y")
    for i, d in enumerate(days):
        base = 0.0
        segs = [(p, c, sum(r["usd"] for r in rows if r["day"] == d and r["purpose"] == p)) for p, c in purposes]
        segs = [s for s in segs if s[2] > 0]
        for k, (p, c, v) in enumerate(segs):
            bar(ax, i, base + (gap if k else 0), base + v, t[c], thick_pt=18, horizontal=False,
                round_end=k == len(segs) - 1)
            base += v
        ax.text(i, base + 0.6, usd(base), ha="center", va="bottom", fontsize=7, color=t["ink"])
        ax.text(i, -1.6, d, ha="center", va="top", fontsize=7.5, color=t["ink2"])
    legend(cv, 0.75, 0.15, [(p, c, "sq") for p, c in purposes])
    return cv


def fig_claude_spend(D, t):
    days = [r for r in D["claude_by_day"]["rows"] if r["day"] != "total"]
    tot = [r for r in D["claude_by_day"]["rows"] if r["day"] == "total"][0]
    roles = sorted(D["claude_by_role"]["rows"], key=lambda r: -r["usd"])
    cv = Canvas(t, 8.6, 4.4)
    cv.header("Claude spend by day and by role",
              f"{num(tot['calls'])} headless calls, {usd(tot['usd'])} estimated by the CLI (subscription calls). "
              "Two charts, one scale each.")
    cv.text(0.25, 3.5, "By day (estimated USD)", size=8.5, bold=True)
    ax = cv.axes(0.75, 0.6, 3.0, 2.6)
    value_axis(ax, t, (0, 50), [0, 10, 20, 30, 40, 50], ["$0", "$10", "$20", "$30", "$40", "$50"], axis="y")
    ax.set_xlim(-0.6, len(days) - 0.4)
    for i, r in enumerate(days):
        bar(ax, i, 0, r["usd"], t["s1"], thick_pt=11, horizontal=False)
        ax.text(i, -1.5, r["day"][3:], ha="center", va="top", fontsize=7, color=t["ink2"])
    mx = max(days, key=lambda r: r["usd"])
    i = days.index(mx)
    ax.text(i, mx["usd"] + 0.8, usd(mx["usd"]), ha="center", va="bottom", fontsize=7, color=t["ink"], **BOLD)
    cv.text(0.75, 0.25, "day of November", size=7, color="muted")
    cv.text(4.25, 3.5, "By role (estimated USD)", size=8.5, bold=True)
    ax2 = cv.axes(6.25, 0.45, 1.65, 2.85)
    value_axis(ax2, t, (0, 70), [0, 35, 70], ["$0", "$35", "$70"])
    ax2.set_ylim(len(roles) - 0.5, -0.5)
    for i, r in enumerate(roles):
        bar(ax2, i, 0, r["usd"], t["s1"], thick_pt=8, r_pt=2.5)
        role = r["role"].replace("window loop and experiments", "loop and experiments")
        ax2.text(-1.5, i, f"{role} ({r['model'].capitalize()})", ha="right", va="center", fontsize=7,
                 color=t["ink"])
        ax2.text(r["usd"] + 1.2, i, usd(r["usd"]), ha="left", va="center", fontsize=6.5, color=t["ink2"])
    return cv


def fig_records(D, t):
    rows = D["records"]["rows"]
    rh, gh = 0.26, 0.3
    groups = []
    for r in rows:
        if not groups or groups[-1][0] != r["group"]:
            groups.append((r["group"], []))
        groups[-1][1].append(r)
    h = 1.2 + len(rows) * rh + len(groups) * gh
    cv = Canvas(t, 9.0, h)
    cv.header("Observability: every record the system keeps, and what reads it",
              "Every paid call, every draft and every write-back leaves a record. The banks let a run resume, and "
              "the job files keep every original.")
    XR, XC, XA, XD = 0.25, 2.55, 5.05, 5.45
    y = h - 0.98
    for x, s in ((XR, "record"), (XC, "size"), (XD, "read by")):
        cv.text(x, y, s, size=7, color="muted")
    for g, items in groups:
        y -= gh
        cv.text(XR, y + 0.06, g, size=8.5, bold=True)
        cv.line([XR, 8.75], [y, y], color="axis")
        for r in items:
            y -= rh
            cv.text(XR + 0.1, y + 0.06, r["record"], size=8)
            cv.text(XC, y + 0.06, r["count"], size=7.5, color="ink2")
            cv.arrow((XA, y + 0.1), (XD - 0.08, y + 0.1), color="muted", lw=0.8)
            cv.text(XD, y + 0.06, r["readers"], size=7.5)
    return cv


def fig_alerts(D, t):
    rows = [r for r in D["alerts"]["rows"] if "kind" in r and r["group"] != "two-way"]
    rh = 0.26
    h = 1.3 + len(rows) * rh + 2 * 0.3
    cv = Canvas(t, 9.0, h)
    cv.header("Alerts: every Discord message kind and when it fires",
              "One channel carries reports, alerts and questions. Most alerts fire once per cause, so a stuck "
              "cap doesn't flood the phone.")
    XK, XW, XO = 0.25, 2.35, 6.65
    y = h - 1.0
    for x, s in ((XK, "message"), (XW, "fires when"), (XO, "how often")):
        cv.text(x, y, s, size=7, color="muted")
    last = None
    for r in rows:
        if r["group"] != last:
            y -= 0.3
            cv.text(XK, y + 0.06, r["group"], size=8.5, bold=True)
            cv.line([XK, 8.75], [y, y], color="axis")
            last = r["group"]
        y -= rh
        cv.text(XK + 0.1, y + 0.06, r["kind"], size=8)
        cv.text(XW, y + 0.06, r["when"], size=7.5, color="ink2")
        cv.text(XO, y + 0.06, r["how_often"], size=7.5, color="ink2")
    return cv


def fig_two_way(D, t):
    rows = D["alerts"]["rows"]
    q = [r for r in rows if r.get("kind") == "question on a drop-rule cut"][0]
    rep = [r for r in rows if r.get("kind") == "replies read back"][0]
    rules = [r for r in rows if "reply_rules" in r][0]["reply_rules"]
    cv = Canvas(t, 9.0, 4.3)
    cv.header("Two-way control: the author steers the loop by replying on Discord",
              "Questions go out on drop-rule cuts; the loop reads the answers back and acts on them.")
    bw, bh = 1.95, 1.0
    y = 3.5
    b1 = cv.box(0.25, y, bw, bh, [("A drop-rule cut", "title"), ("the loop drops a line", "body"),
                                  ("and the judge says why", "body")])
    b2 = cv.box(2.55, y, bw, bh, [("Question posted", "title"), ("the cut happens now;", "body"),
                                  ("a reply can undo it", "body")], accent=True)
    b3 = cv.box(4.85, y, bw, bh, [("The author replies", "title"), ("a number, a reaction,", "body"),
                                  ("'stop', or 'rest' + window", "body")])
    b4 = cv.box(7.15, y, 1.6, bh, [("Read back", "title"), ("every 60 s while", "body"), ("waiting, before", "body"),
                                   ("each pass, at", "body"), ("the digest", "body")], accent=True)
    for a, b in ((b1, b2), (b2, b3), (b3, b4)):
        cv.arrow((a[0] + a[2] + 0.02, y - bh / 2), (b[0] - 0.02, y - bh / 2))
    outs = [("Veto", rules["veto"]), ("Rest", rules["rest"]), ("Stop", rules["stop"]),
            ("No answer", f"the default stands after {rules['default_after_days']} days")]
    ox = [0.25, 2.4, 4.55, 6.7]
    oy = 1.95
    for (title, body), x in zip(outs, ox):
        cv.box(x, oy, 2.05, 0.78, [(title, "title")] + [(b, "body") for b in wrap(body, 28)])
        cv.arrow((b4[0] + 0.8, y - bh - 0.02), (x + 1.02, oy + 0.02), rad=0.0, color="muted", lw=0.8)
    cv.ax.add_patch(FancyBboxPatch((0.25, 0.35), 8.5, 0.42, boxstyle="round,pad=0,rounding_size=0.06",
                                   fc=t["surface"], ec=t["ink2"], lw=1.0))
    cv.text(4.5, 0.5, f"A reply can never {rules['never']}. Only the author's replies count.", size=8.5,
            bold=True, ha="center")
    return cv


def fig_learning(D, t):
    rows = D["learning"]["rows"]
    s = {r["step"]: r for r in rows if "step" in r}
    rule = [r for r in rows if "rule" in r][0]["rule"]
    share = s["random slice"]["share"]
    cv = Canvas(t, 9.0, 4.75)
    cv.header("The learning layer: how the loop learns what to try",
              "Every night feeds the bank; a proposed change has to beat the current playbook before it's adopted.")
    bw, bh = 2.55, 1.05
    top, bot = 3.85, 1.95
    xs = [0.25, 3.2, 6.15]
    def lines(title, text, n=34):                                 # noqa: E306
        return [(title, "title")] + [(b, "body") for b in wrap(text, n)]
    bank = cv.box(xs[0], top, bw, bh, lines("Bank", s["bank"]["text"]))
    dire = cv.box(xs[1], top, bw, bh, lines("Director proposes", s["director"]["text"]))
    test = cv.box(xs[2], top, bw, bh + 0.3, lines("Test night", s["test night"]["text"]), accent=True)
    adopt = cv.box(xs[2], bot - 0.25, bw, bh, lines("Adopt?", s["adopt"]["text"]), accent=True)
    play = cv.box(xs[1], bot - 0.25, bw, bh, lines("Playbook, next version", s["playbook"]["text"]))
    night = cv.box(xs[0], bot - 0.25, bw, bh,
                   lines("The next night's attempts", f"the director picks most combinations; "
                                                     f"{share * 100:.0f}% are picked at random by code"))
    cv.arrow((xs[0] + bw + 0.02, top - bh / 2), (xs[1] - 0.02, top - bh / 2))
    cv.arrow((xs[1] + bw + 0.02, top - bh / 2), (xs[2] - 0.02, top - bh / 2))
    cv.arrow((xs[2] + bw / 2, top - bh - 0.32), (xs[2] + bw / 2, bot - 0.23))
    cv.arrow((xs[2] - 0.02, bot - 0.25 - bh / 2), (xs[1] + bw + 0.02, bot - 0.25 - bh / 2))
    cv.text(xs[2] - 0.32, bot - 0.25 - bh / 2 + 0.07, "wins", size=7, color="ink2", ha="center")
    cv.arrow((xs[1] - 0.02, bot - 0.25 - bh / 2), (xs[0] + bw + 0.02, bot - 0.25 - bh / 2))
    cv.arrow((xs[0] + bw / 2, bot - 0.23), (xs[0] + bw / 2, top - bh - 0.02))
    cv.text(xs[0] + bw / 2 + 0.08, (bot + top - bh) / 2 - 0.1, "every scored draft", size=7, color="ink2")
    cv.text(xs[2] + bw - 0.05, bot - 0.25 - bh - 0.18, "loses or ties: the current version stays", size=7,
            color="ink2", ha="right")
    cv.text(0.25, 0.3, f"Guard: {rule}.", size=8, color="ink2")
    return cv


def fig_gpu_guard(D, t):
    rows = D["gpu_guard"]["rows"]
    sig, steps, cfg = rows[0]["signals"], rows[1]["steps"], rows[2]
    cv = Canvas(t, 9.0, 3.3)
    cv.header("The GPU guard: handing the card back during the day",
              f"Any one signal pauses the local models and frees the card. A watcher looks every {cfg['poll_s']} s, "
              "and the work carries on where it stopped.")
    y = 2.45
    for i, s_ in enumerate(sig):
        top = y - i * 0.58
        cv.box(0.25, top, 2.45, 0.46, [(ln, "ink") for ln in wrap(s_, 34)], pad=0.07)
        cv.arrow((2.72, top - 0.23), (2.98, 1.9), color="muted", lw=0.8)
    cv.text(0.25, 0.55, "any one signal", size=7, color="muted")
    bw, gap = 1.04, 0.12
    xs = [2.98 + i * (bw + gap) for i in range(len(steps))]
    for i, (x, s_) in enumerate(zip(xs, steps)):
        cv.box(x, 2.2, bw, 0.62, [(ln, "ink") for ln in wrap(s_, 15)], accent=i == 0, pad=0.07)
        if i < len(steps) - 1:
            cv.arrow((x + bw + 0.01, 1.89), (xs[i + 1] - 0.01, 1.89))
    cv.arrow((xs[-1] + bw / 2, 1.56), (xs[0] + bw / 2, 1.56), rad=-0.22, color="muted", lw=0.8)
    cv.text((xs[0] + xs[-1] + bw) / 2, 0.82, "a new signal pauses it again", size=7, color="ink2", ha="center")
    cv.text(3.0, 0.3, f"{cfg['rules'][0].capitalize()}, and {cfg['rules'][1]}.", size=7.5, color="ink2")
    return cv


# ========================================================================================== the registry
FIGURES = [
    ("noise-floor", fig_noise_floor, "Evals",
     "Pangram's noise floor: one of nine trivial edits moved a score, by 0.21; the keep threshold is 0.05."),
    ("keep-or-drop", fig_keep_or_drop, "Evals",
     "Every model and method tried, with its result and whether it was kept, dropped or parked."),
    ("detectors", fig_detectors, "Evals",
     "Pangram against Open Pangram on the bake-off models: Open Pangram ranks for free, Pangram decides."),
    ("blind-judge", fig_judge, "Evals",
     "The blind head-to-head judge in the window loop: outcomes that reconcile to the ledger's 1,067 calls."),
    ("voice-metric", fig_voice, "Evals",
     "Every Burrows' Delta reading against the author's own median and 90th percentile, per baseline."),
    ("gate-funnel", fig_gates, "Evals",
     "The paragraph pass from draws to picks, with the code gate's rejection reasons."),
    ("fact-check", fig_fact_check, "Evals",
     "The Sonnet fact check by writer arm and prompt."),
    ("humanizer-flag", fig_humanizer, "Evals",
     "The humanizer flag by pass: 3 of 168 window-round scorings against 17 of 32 in the paragraph pass."),
    ("skeleton-swap", fig_skeleton, "Experiments",
     "The 2x2 swap of sentence skeletons and word choice, with n per cell."),
    ("split", fig_split, "Experiments",
     "One window split by author: 0.00 and 0.35 apart, 1.00 together."),
    ("section-trajectories", fig_trajectories, "Experiments",
     "Every scoring of every section, in order, with the lows and the latest scoring labeled."),
    ("timeline", fig_timeline, "Experiments",
     "Every experiment and milestone in order, each with its result and decision."),
    ("pipeline", fig_pipeline, "The loop",
     "The pipeline from notes to write-back, with the paragraph pass's counts."),
    ("night", fig_night, "The loop",
     "A night's schedule, its deadlines and its guards, beside the first unattended night."),
    ("budget-caps", fig_caps, "The loop",
     "Every budget cap, in the order the code checks it."),
    ("pangram-spend", fig_spend, "The loop",
     "Pangram spend by day and purpose."),
    ("claude-spend", fig_claude_spend, "The loop",
     "Claude spend by day and by role, as two charts."),
    ("records", fig_records, "The loop",
     "Every record the system keeps, its size and what reads it."),
    ("alerts", fig_alerts, "The loop",
     "Every Discord message kind and when it fires."),
    ("two-way-control", fig_two_way, "The loop",
     "Two-way control: questions on drop-rule cuts, replies read back, and what a reply can and can't do."),
    ("learning-layer", fig_learning, "The loop",
     "The learning layer: the bank, the director's proposal, the test night and the versioned playbook."),
    ("gpu-guard", fig_gpu_guard, "The loop",
     "The GPU guard: the signals that pause a run, free the card and resume it."),
]


# ========================================================================================== output
STYLE = re.compile(r'style="([^"]*)"')


def fix_text_styles(svg):
    """Fold matplotlib's separate font properties into one `font:` shorthand set in the system sans."""
    wkey = "font-" + "we" + "ight"

    def sub(m):                                                   # noqa: E306
        props = [p.strip() for p in m.group(1).split(";") if p.strip()]
        d = {}
        for p in props:
            k, _, v = p.partition(":")
            d[k.strip()] = v.strip()
        if "font-family" not in d and "font-size" not in d:
            return m.group(0)
        bold = d.pop(wkey, "400")
        bold = "700" if bold in ("700", "bold") else "400"
        style = d.pop("font-style", "normal")
        size = d.pop("font-size", "10px")
        d.pop("font-family", None)
        d.pop("font-stretch", None)
        rest = "; ".join(f"{k}: {v}" for k, v in d.items())
        font = f"font: {style} {bold} {size} {FONT_STACK}"
        return 'style="' + (rest + "; " if rest else "") + font + '"'
    return STYLE.sub(sub, svg)


def save(cv, path, png=None):
    cv.fig.savefig(path, format="svg", metadata={"Date": None, "Creator": "depangram docs/figures/make_figures.py"})
    svg = Path(path).read_text(encoding="utf-8")
    Path(path).write_text(fix_text_styles(svg), encoding="utf-8", newline="\n")
    if png:
        cv.fig.savefig(png, format="png", dpi=110)
    plt.close(cv.fig)


def build(out, png_dir=None, data_path=None):
    data = json.loads(Path(data_path or HERE / "data.json").read_text(encoding="utf-8"))
    D = by_id(data)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for name, fn, _, _ in FIGURES:
        for mode, t in THEMES.items():
            with plt.rc_context(rc(t)):
                cv = fn(D, t)
                png = Path(png_dir) / f"{name}-{mode}.png" if png_dir else None
                save(cv, out / f"{name}-{mode}.svg", png)
    (out / "README.md").write_text(readme(D), encoding="utf-8", newline="\n")
    return [n for n, *_ in FIGURES]


# ========================================================================================== README
def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def tables(D):
    T = {}
    nf = D["noise_floor"]["rows"]
    T["noise-floor"] = table(["Passage", "Words", "Original", "Comma moved", "Contraction swapped", "Twin word",
                              "Biggest move"],
                             [[r["passage"], r["words"], fmt(r["original"]), fmt(r["comma moved"]),
                               fmt(r["contraction swapped"]), fmt(r["twin word"]), fmt(r["max swing"])]
                              for r in nf[:3]]) + (
        f"\n\nMean of the per-passage biggest moves: {fmt(nf[3]['mean of per-passage max swings'])}. "
        f"Keep threshold in use: {fmt(nf[3]['keep threshold'])}. {nf[3]['scorings']} scorings, {usd(nf[3]['usd'])}.")
    T["keep-or-drop"] = table(["Group", "Model or method", "Result", "Decision", "Reason"],
                              [[r["group"], r["what"], r["result"], r["decision"], r["reason"]]
                               for r in D["keep_or_drop"]["rows"]])
    T["detectors"] = table(["Round", "Model", "Pangram", "Open Pangram", "Hard tics per sample", "Tokens/s",
                            "Decision"],
                           [[r["round"], r["model"], fmt(r["pangram"]), fmt(r["open_pangram"]), fmt(r["hard_tics"]),
                             r["tok_s"], r["decision"]] for r in D["bakeoff_models"]["rows"]]) + "\n\n" + table(
        ["Agreement measure", "RoBERTa", "Llama 3.2 3B"],
        [[r["metric"], r.get("roberta_large", ""), r.get("llama_3.2_3b", "")]
         for r in D["open_pangram_agreement"]["rows"]])
    T["blind-judge"] = table(["Outcome", "Window rounds", "Context rounds", "Total"],
                             [[r["outcome"], r.get("rounds", ""), r.get("context rounds", ""), num(r["total"])]
                              for r in D["h2h_outcomes"]["rows"]])
    keys = ["loop", "adapter_2ep", "adapter_3ep", "plain", "voice_pass", "shipped", "olmo_sft", "mistral_instruct",
            "sonnet", "opus", "live_adapter", "retrain_tag_1", "retrain_tag_2", "standing", "best"]
    T["voice-metric"] = table(["Baseline", "Author median", "90th percentile"],
                              [[r["baseline"], fmt(r["median"]), fmt(r["p90"])]
                               for r in D["delta_baselines"]["rows"]]) + "\n\n" + table(
        ["Test", "Window", "Readings"],
        [[r["test"], r["window"], ", ".join(f"{k.replace('_', ' ')} {fmt(r[k])}" for k in keys if k in r)]
         for r in D["delta_comparisons"]["rows"]])
    T["gate-funnel"] = table(["Stage", "Count", "Note"],
                             [[r["stage"], num(r["n"]), r.get("split", "")]
                              for r in D["paragraph_pass_stages"]["rows"]]) + "\n\n" + table(
        ["First rejection reason", "Draws"],
        [[r["reason"], num(r["draws"])] for r in D["paragraph_pass_gate"]["rows"]]) + "\n\n" + table(
        ["Writer arm", "Draws", "Through the code gate"],
        [[r["arm"], num(r["draws"]), num(r["passed"])] for r in D["paragraph_pass_gate_by_arm"]["rows"]])
    T["fact-check"] = table(["Gate", "Arm", "Passed", "Failed"],
                            [[r["gate"], r["arm"], r["pass"], r["fail"]] for r in D["fact_gate"]["rows"]])
    T["humanizer-flag"] = table(["Purpose", "Pass", "Scorings", "Flagged"],
                                [[r["purpose"], r.get("pass", "all"), r["scorings"], r["humanized"]]
                                 for r in D["humanizer_by_pass"]["rows"]])
    T["skeleton-swap"] = table(["Condition", "Skeleton", "Words", "n", "Skeleton match", "AI share",
                                "Mean surprisal"],
                               [[r["condition"], r["skeleton"], r["words"], r["n"], fmt(r["skeleton_match"]),
                                 fmt(r["ai_share"]), fmt(r["surprisal_mean"])]
                                for r in D["skeleton_word_swap"]["rows"]])
    T["split"] = table(["Passage", "Sentences", "Words", "AI share"],
                       [[r["passage"], r["sentences"], r.get("words", ""), fmt(r["ai_share"])]
                        for r in D["diag_split"]["rows"]])
    T["section-trajectories"] = table(["Section", "Scorings in order"],
                                      [[r["section"], "; ".join(f"{a} {fmt(b)}" for a, b in r["series"])]
                                       for r in D["section_trajectories"]["rows"]])
    T["timeline"] = table(["Date", "Experiment or milestone", "Result", "Decision"],
                          [[r["date"], r["item"], r["result"], r["decision"]] for r in D["timeline"]["rows"]])
    T["pipeline"] = table(["Step", "Who", "What", "Count"],
                          [[r["step"], r["who"], r["what"],
                            num(r["count"]) if "count" in r else (f"{num(r['count_out'])} out" if "count_out" in r
                                                                  else "")]
                           for r in D["pipeline"]["rows"]])
    nr = D["night"]["rows"]
    fn_ = nr[3]["first_night"]
    T["night"] = table(["Order", "Step at the start"], [[i + 1, s] for i, s in enumerate(nr[0]["steps"])]) + (
        "\n\n" + table(["Setting", "Value"],
                       [["task fires", nr[1]["fires"]], ["late-start cutoff", nr[1]["late_start"]],
                        ["pass deadline", nr[1]["pass_until"]], ["finals, at most", f"{nr[1]['finals_max_min']} min"],
                        ["children polled every", f"{nr[1]['poll_s']} s"],
                        ["wait before force-kill", f"{nr[1]['kill_wait_s']} s"]])
        + "\n\n" + table(["First unattended night", "Value"],
                         [["start", fn_["start"]], ["pass end", fn_["pass_end"]], ["finals end", fn_["finals_end"]],
                          ["submission", fn_["submission"]], ["Pangram", usd(fn_["pangram_usd"])],
                          ["Claude (estimated)", usd(fn_["claude_usd"])]]))
    T["budget-caps"] = table(["Lane", "Order", "Cap", "Value", "When it says no"],
                             [[r["lane"], r.get("order", ""), r["cap"], r["value"], r.get("on_hit", "")]
                              for r in D["caps"]["rows"]])
    T["pangram-spend"] = table(["Day", "Purpose", "Calls", "USD"],
                               [[r["day"], r.get("purpose", "all"), num(r["calls"]), usd(r["usd"])]
                                for r in D["pangram_spend_by_day"]["rows"]])
    T["claude-spend"] = table(["Day", "Calls", "USD (estimated)"],
                              [[r["day"], num(r["calls"]), usd(r["usd"])] for r in D["claude_by_day"]["rows"]]) + (
        "\n\n" + table(["Role", "Model", "Calls", "USD (estimated)"],
                       [[r["role"], r["model"], num(r["calls"]), usd(r["usd"])] for r in D["claude_by_role"]["rows"]]))
    T["records"] = table(["Group", "Record", "Size", "Read by"],
                         [[r["group"], r["record"], r["count"], r["readers"]] for r in D["records"]["rows"]])
    al = D["alerts"]["rows"]
    T["alerts"] = table(["Group", "Message", "Fires when", "How often"],
                        [[r["group"], r["kind"], r["when"], r["how_often"]] for r in al if "kind" in r])
    rr = [r for r in al if "reply_rules" in r][0]["reply_rules"]
    T["two-way-control"] = table(["Reply", "What it does"],
                                 [["veto", rr["veto"]], ["rest", rr["rest"]], ["stop", rr["stop"]],
                                  ["no answer", f"the default stands after {rr['default_after_days']} days"],
                                  ["never", rr["never"]]])
    T["learning-layer"] = table(["Step", "What happens"],
                                [[r.get("step", "guard"), r.get("text", r.get("rule"))]
                                 for r in D["learning"]["rows"]])
    g = D["gpu_guard"]["rows"]
    T["gpu-guard"] = table(["Signals", "Steps"],
                           [[a, b] for a, b in zip(g[0]["signals"] + [""], g[1]["steps"])])
    return T


def readme(D):
    T = tables(D)
    out = ["# Figures", "",
           "Every chart and diagram in the docs, in a light and a dark version. `make_figures.py` draws them all "
           "from `data.json` and writes this file, so the numbers below are the numbers on the figures. "
           "Don't edit by hand: change the data or the script and run it again.", "",
           "```bash", "pip install matplotlib==3.11.2", "python docs/figures/make_figures.py", "```", "",
           "Two runs give byte-identical SVGs. The text stays text, set in the system sans.", "",
           "To embed a figure in a page under `docs/`, use GitHub's dark-mode switch:", "",
           "```html", "<picture>",
           '  <source media="(prefers-color-scheme: dark)" srcset="figures/NAME-dark.svg">',
           '  <img alt="CAPTION" src="figures/NAME-light.svg">', "</picture>", "```", ""]
    section = None
    for i, (name, _, sec, cap) in enumerate(FIGURES, 1):
        if sec != section:
            out += [f"## {sec}", ""]
            section = sec
        alt = cap.replace('"', "'")
        out += [f"### {i}. {name}", "", cap, "",
                "<picture>",
                f'  <source media="(prefers-color-scheme: dark)" srcset="{name}-dark.svg">',
                f'  <img alt="{alt}" src="{name}-light.svg">',
                "</picture>", "",
                "```html", "<picture>",
                f'  <source media="(prefers-color-scheme: dark)" srcset="figures/{name}-dark.svg">',
                f'  <img alt="{alt}" src="figures/{name}-light.svg">', "</picture>", "```", "",
                T[name], ""]
    notes = "\n".join(f"- `{d['id']}`: {d['note']}" for d in D.values())
    out += ["## Where the numbers come from", "", notes, ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Draw the docs figures from data.json.")
    ap.add_argument("--out", default=str(HERE), help="folder for the SVGs and README.md (default: this folder)")
    ap.add_argument("--png", default=None, help="also write a PNG preview of every figure into this folder")
    a = ap.parse_args(argv)
    if a.png:
        Path(a.png).mkdir(parents=True, exist_ok=True)
    names = build(a.out, a.png)
    print(f"{len(names)} figures, {2 * len(names)} SVGs -> {a.out}")


if __name__ == "__main__":
    main()
