"""Shared figure style for the TIB analysis report (paper quality).

Usage:
    import figstyle as fs
    fs.setup()                 # paper: serif (Latin Modern), 8-9 pt, vector PDF with embedded fonts
    fs.setup("slides")         # sans (Roboto), slightly larger, for slides
    fig, ax = plt.subplots(figsize=(fs.COL, 2.2))
    ...
    fs.save(fig, "fig01_forest")   # -> docs/figures/report/fig01_forest.{pdf,png}

Fixed colours: every figure uses the same colour for an arm and the same order/names for models.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import seaborn as sns  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUTDIR = ROOT / "docs" / "figures" / "report"

# Column widths (inches) for NeurIPS / ICML style papers.
COL = 3.25
DCOL = 6.75

# ── arms ────────────────────────────────────────────────────────────────────
ARM_COLOR = {
    "stock": "#7f7f7f",               # grey
    "claims_after": "#ff7f0e",        # orange  (check after feedback)
    "claims_no_p": "#e3b178",         # tan     (Bet Card without numbers)
    "commit_before": "#1f77b4",       # blue    (always-on betting)
    "cb_gated": "#17a2a2",            # teal    (on-demand betting)
    "cb_draftfirst": "#5e3c99",       # purple  (draft-first betting)
    # muted baselines
    "conf_gate": "#a59fc9",
    "sc_vote3": "#9fc79a",
    "verify_then_respond": "#d4a3b4",
    "tracker": "#c9c9a3",
    "mediator": "#b3c7d6",
    # E2 aliases
    "standard": "#ff7f0e",            # E2 standard = rates after reading the reply (check-after)
    "bet_first": "#1f77b4",           # E2 bet-first = commits before the reply
    "none": "#bdbdbd",                # E2 blind rating, no reply, no commitment
}
ARM_LABEL = {
    "stock": "Standard agent",
    "claims_after": "Check after feedback",
    "claims_no_p": "Bet Card w/o numbers",
    "commit_before": "Always-on betting",
    "cb_gated": "On-demand betting",
    "cb_draftfirst": "Draft-first betting",
    "conf_gate": "Confidence gate",
    "sc_vote3": "Majority vote of 3",
    "verify_then_respond": "Re-run tests first",
    "standard": "Check after feedback",
    "bet_first": "Bet before reply",
    "none": "Blind rating (no reply)",
}
OURS = ("cb_draftfirst", "commit_before", "cb_gated")

# ── models ──────────────────────────────────────────────────────────────────
MODEL_ORDER = ["W", "F", "M", "S", "G"]
MODEL_LABEL = {"W": "3.5 Flash-Lite", "F": "3.5 Flash", "M": "3.6 Flash", "S": "3.8 Flash", "G": "4.0 Pro"}
MODEL_SHORT = {"W": "3.5 FL", "F": "3.5 F", "M": "3.6 F", "S": "3.8 F", "G": "4.0 Pro"}
MODEL_RANK = {"W": 1, "F": 2, "M": 3, "S": 4, "G": 5}
LABEL_TO_MODEL = {v: k for k, v in MODEL_LABEL.items()}
MODEL_MARKER = {"W": "o", "F": "s", "M": "^", "S": "D", "G": "*"}
# sequential (light -> dark, viridis) so model strength reads at a glance; colour-blind safe
_seq = [matplotlib.colormaps["viridis"](x) for x in (0.82, 0.62, 0.45, 0.28, 0.06)]
MODEL_COLOR = {m: _seq[i] for i, m in enumerate(MODEL_ORDER)}

# ── benchmarks ──────────────────────────────────────────────────────────────
BENCH_LABEL = {
    "swi_mixed": "SWE-Interact, mixed feedback",
    "swi_overlay": "SWE-Interact, pure pushback",
    "ale_e2": "ALE-E2, reviewer pushback",
    "hle_mt": "HLE, 2-turn pushback",
    "fin_mt": "Finance Agent, 2-turn pushback",
    "hle_ss": "HLE, single shot",
    "fin_tools": "Finance Agent, live tools",
    "fin_closed": "Finance Agent, closed book",
    "ds11": "DeepSWE 1.1 (opencode)",
    "ds11m": "DeepSWE 1.1 (mini-swe-agent)",
    "tb4": "Terminal-Bench 4.0",
    "ale_t300": "ALE-Claw, 300 turns",
    "ale_s1": "ALE-Claw, 80 turns",
}

PAPER_RC = {
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "savefig.dpi": 300,
    "figure.dpi": 150,
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.title_fontsize": 7.5,
    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.borderaxespad": 0.3,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 0.9,
    "lines.markersize": 3.5,
    "errorbar.capsize": 0,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.color": "#e6e6e6",
    "grid.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 3,
    "axes.labelpad": 2,
    "hatch.linewidth": 0.5,
}

_MODE = {"mode": "paper"}


def setup(mode: str = "paper") -> None:
    """Apply the shared theme. mode='paper' (serif, Latin Modern) or 'slides' (sans, Roboto, larger)."""
    sns.set_theme(style="ticks", palette="colorblind")
    plt.rcParams.update(PAPER_RC)
    if mode == "slides":
        plt.rcParams.update({"font.family": "sans-serif",
                             "font.sans-serif": ["Roboto", "DejaVu Sans"],
                             "mathtext.fontset": "dejavusans",
                             "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 11,
                             "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9})
    else:
        plt.rcParams.update({"font.family": "serif",
                             "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"]})
    _MODE["mode"] = mode


def save(fig, name: str, outdir: Path | None = None) -> list[Path]:
    """Save ``fig`` as vector PDF (primary) and 300-dpi PNG."""
    out = Path(outdir or OUTDIR)
    out.mkdir(parents=True, exist_ok=True)
    paths = [out / f"{name}.pdf", out / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return paths


def panel_label(ax, s: str, x: float = -0.02, y: float = 1.02) -> None:
    """Bold (a)/(b) panel tag at the top-left corner, outside the axes."""
    ax.text(x, y, s, transform=ax.transAxes, fontweight="bold", ha="right", va="bottom",
            fontsize=plt.rcParams["axes.titlesize"])


def ptitle(ax, tag: str, text: str = "", **kw) -> None:
    """Left-aligned panel header '(a) text' (avoids collisions between tag and a centred title)."""
    kw = {"fontsize": plt.rcParams["axes.titlesize"], "loc": "left", **kw}
    ax.set_title(f"({tag}) {text}".rstrip(), **kw)


def log_tokens_axis(ax) -> None:
    """Readable log x-axis for token counts. Call after plotting: picks 2-4 'nice' ticks inside the current
    limits (1-2-5 x 10^k, falling back to 1-1.5-2-3-5-7 x 10^k) and labels them 3k / 2M."""
    import math
    from matplotlib import ticker

    def _fmt(v, _pos=None):
        if v >= 1e6:
            return f"{v / 1e6:g}M"
        if v >= 1e3:
            return f"{v / 1e3:g}k"
        return f"{v:g}"
    lo, hi = ax.get_xlim()
    ticks = []
    for subs in ((1, 2, 5), (1, 1.5, 2, 3, 5, 7), (1, 1.2, 1.4, 1.6, 1.8, 2, 2.5, 3, 4, 5, 6, 7, 8, 9)):
        ticks = [s * 10 ** k for k in range(int(math.floor(math.log10(lo))) - 1, int(math.ceil(math.log10(hi))) + 1)
                 for s in subs if lo <= s * 10 ** k <= hi]
        if len(ticks) >= 2:
            break
    if len(ticks) > 4:
        ticks = ticks[:: int(math.ceil(len(ticks) / 4))]
    ax.xaxis.set_major_locator(ticker.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(_fmt))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.set_xlim(lo, hi)


def zero_line(ax, axis: str = "x", **kw) -> None:
    kw = {"color": "#555555", "lw": 0.6, "zorder": 0.5, **kw}
    (ax.axvline if axis == "x" else ax.axhline)(0, **kw)


def arm_handles(arms, marker="o", ls="none", ms=4):
    from matplotlib.lines import Line2D
    return [Line2D([], [], color=ARM_COLOR[a], marker=marker, ls=ls, ms=ms, label=ARM_LABEL[a]) for a in arms]
