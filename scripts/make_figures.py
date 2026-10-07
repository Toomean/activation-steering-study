# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["matplotlib==3.11.2"]
# ///
"""Rebuild the three study figures from checksum-verified released aggregates.

Plotted intervals are published inputs, not newly computed estimates. The pipeline
schematic reproduces the accepted methods layout. Run from the repository root.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
from fractions import Fraction
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.text import Text
from matplotlib.textpath import TextToPath

from report_tables import verify_inputs

OUT_DIR = Path("build/figures")
DEVELOPMENT_CSV = Path("results/development/development.csv")
DOSE_SELECTION_CSV = Path("results/development/dose-selection.csv")
FINAL_PLOT_CSV = Path("results/positive/final-plot-data.csv")
SOURCE_SHA256 = {
    Path(entry["path"]): entry["projection_sha256"]
    for entry in verify_inputs()["files"]
    if entry["path"] in {str(DEVELOPMENT_CSV), str(DOSE_SELECTION_CSV), str(FINAL_PLOT_CSV)}
}

CM = 1 / 2.54  # inches per centimetre
WIDTH_CM = 16.0  # A4 text width between 2.5 cm margins; the PDF places figures at 100%
MIN_FONT_PT = 8.0
PNG_DPI = 300

# Neutral inks and the first three categorical slots of the dataviz reference palette; the three
# slots pass the all-pairs colour-vision-deficiency check on a white surface. Marker shapes and
# line styles repeat every identity so the figures also read in greyscale.
INK, INK_SECONDARY, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BOX_FILL, BOX_EDGE = "#f4f4f2", "#b9b8b1"
DIRECTIONS = ("pooled", "cyber_intrusion", "dangerous_substances", "disinformation")
DOMAINS = DIRECTIONS[1:]
LABEL = {
    "pooled": "Pooled",
    "cyber_intrusion": "Cyber intrusion",
    "dangerous_substances": "Dangerous substances",
    "disinformation": "Disinformation",
}
# The pooled direction is the fixed reference, so it takes a neutral ink rather than a hue.
COLOUR = {
    "pooled": "#3a3936",
    "cyber_intrusion": "#2a78d6",
    "dangerous_substances": "#eb6834",
    "disinformation": "#1baf7a",
}
MARKER = {"pooled": "D", "cyber_intrusion": "o", "dangerous_substances": "s", "disinformation": "^"}
LINESTYLE = {"cyber_intrusion": "-", "dangerous_substances": (0, (4.5, 2)), "disinformation": (0, (1, 1.6))}
# Horizontal offsets on the log2 dose axis, in log2 units, so coincident points stay visible.
DODGE_LOG2 = {"pooled": -0.15, "cyber_intrusion": -0.05, "dangerous_substances": 0.05, "disinformation": 0.15}
COMPARATORS = ("baseline", "R42", "R43")
COMPARATOR_LABEL = {"baseline": "Real − baseline", "R42": "Real − R42", "R43": "Real − R43"}
COMPARATOR_MARKER = {"baseline": "o", "R42": "s", "R43": "^"}
COMPARATOR_OFFSET = {"baseline": 0.24, "R42": 0.0, "R43": -0.24}  # within a direction row, top to bottom

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "mathtext.fontset": "custom",
        "mathtext.rm": "DejaVu Sans",
        "mathtext.it": "DejaVu Sans:italic",
        "mathtext.bf": "DejaVu Sans:bold",
        "mathtext.cal": "DejaVu Sans:italic",  # unused; avoids a lookup of an absent cursive family
        "svg.fonttype": "path",  # glyphs as outlines: identical rendering without the fonts installed
        "svg.hashsalt": "a3-report-figures",  # deterministic element ids across rebuilds
    }
)


def _read_rows(path: Path) -> list[dict[str, str]]:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256[path]:
        sys.exit(f"{path}: SHA-256 {digest} differs from the frozen digest {SOURCE_SHA256[path]}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _save(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [OUT_DIR / f"{stem}.svg", OUT_DIR / f"{stem}.png"]
    fig.savefig(paths[0], metadata={"Date": None})
    fig.savefig(paths[1], dpi=PNG_DPI, metadata={"Software": None})
    return paths


def _check_fonts_and_width(fig: plt.Figure, name: str) -> None:
    fig.canvas.draw()
    width_cm = fig.get_figwidth() / CM
    assert abs(width_cm - WIDTH_CM) < 1e-9, f"{name}: width {width_cm:.3f} cm, expected {WIDTH_CM} cm"
    figure_box = fig.bbox
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        assert text.get_fontsize() >= MIN_FONT_PT, f"{name}: {text.get_text()!r} is {text.get_fontsize()} pt"
        extent = text.get_window_extent()
        inside = extent.x0 >= figure_box.x0 - 0.5 and extent.x1 <= figure_box.x1 + 0.5
        assert inside and extent.y0 >= figure_box.y0 - 0.5 and extent.y1 <= figure_box.y1 + 0.5, (
            f"{name}: {text.get_text()!r} extends beyond the figure"
        )


# ---------------------------------------------------------------- Figure 4.1: development


def _development_data() -> tuple[dict[tuple[str, float], tuple[int, float]], dict[str, float]]:
    """Return {(direction, dose): (refusal change in requests of 63, MMLU change in pp)} and frozen doses."""
    rows = [row for row in _read_rows(DEVELOPMENT_CSV) if row["role"] == "development"]
    (baseline,) = [row for row in rows if row["control"] == "baseline"]
    points = {}
    for row in rows:
        if row["control"] != "real":
            continue
        # Every development group is one request with equal weight, so the paired change in
        # requests equals the difference between refusal counts.
        requests = int(row["refusal_count"]) - int(baseline["refusal_count"])
        points[(row["target"], float(row["alpha"]))] = (requests, float(row["mmlu_paired_change"]) * 100)
    frozen = {
        row["target"]: float(row["alpha"]) for row in _read_rows(DOSE_SELECTION_CSV) if row["selected"] == "True"
    }
    return points, frozen


def _dodged(direction: str, dose: float) -> float:
    return dose * 2 ** DODGE_LOG2[direction]


def draw_development() -> plt.Figure:
    points, frozen = _development_data()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_CM * CM, 7.4 * CM), layout="constrained")
    panels = (
        (axes[0], 0, "(a) Refusal proxy, 63 requests", "Change from baseline\n(requests of 63)"),
        (axes[1], 1, "(b) MMLU accuracy, 285 questions", "Change from baseline\n(percentage points)"),
    )
    for ax, index, title, ylabel in panels:
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2], ["+0.5", "+1", "+2"])
        ax.minorticks_off()
        ax.set_xlim(2**-1.4, 2**1.4)
        ax.set_xlabel("Dose α (dimensionless, log scale)")
        ax.set_ylabel(ylabel)
        ax.set_title(title.replace(", ", ",\n"), fontsize=9, color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        ax.axhline(0, color=INK_SECONDARY, linewidth=0.8, zorder=1)
        for direction in DIRECTIONS:
            doses = sorted(dose for target, dose in points if target == direction)
            values = [points[(direction, dose)][index] for dose in doses]
            colour, marker = COLOUR[direction], MARKER[direction]
            if direction != "pooled":
                ax.plot(
                    [_dodged(direction, dose) for dose in doses],
                    values,
                    color=colour,
                    linestyle=LINESTYLE[direction],
                    linewidth=1.3,
                    marker=marker,
                    markersize=5.5,
                    markerfacecolor="white",
                    markeredgecolor=colour,
                    markeredgewidth=1.2,
                    zorder=3,
                    gid=f"development-{index}-{direction}",
                )
            dose = frozen[direction]
            ax.plot(
                [_dodged(direction, dose)],
                [points[(direction, dose)][index]],
                linestyle="none",
                marker=marker,
                markersize=5.5,
                markerfacecolor=colour,
                markeredgecolor=colour,
                markeredgewidth=1.2,
                zorder=4,
                gid=f"development-{index}-{direction}-frozen",
            )
    axes[0].set_ylim(-4, 66)
    axes[1].set_ylim(-9, 5)
    handles = [
        Line2D([], [], color=COLOUR["pooled"], marker="D", markersize=5.5, linestyle="none", label="Pooled (fixed at +1)")
    ]
    handles += [
        Line2D(
            [],
            [],
            color=COLOUR[direction],
            linestyle=LINESTYLE[direction],
            linewidth=1.3,
            marker=MARKER[direction],
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label=LABEL[direction],
        )
        for direction in DOMAINS
    ]
    handles += [
        Line2D([], [], color=MUTED, marker="o", markersize=5.5, linestyle="none", label="Frozen dose (filled)"),
        Line2D(
            [],
            [],
            color=MUTED,
            marker="o",
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=1.2,
            linestyle="none",
            label="Other tested dose (open)",
        ),
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, columnspacing=1.6)
    return fig


def check_development(fig: plt.Figure) -> int:
    """Compare the drawn points with exact fractions from the CSVs and the released development data."""
    rows = [row for row in _read_rows(DEVELOPMENT_CSV) if row["role"] == "development" and row["control"] == "real"]
    artists = {line.get_gid(): line for line in fig.findobj(Line2D) if line.get_gid()}
    expected: dict[tuple[str, float], tuple[Fraction, Fraction]] = {}
    for row in rows:
        refusal = Fraction(row["refusal_paired_change_fraction"]) * int(row["harmless_groups"])
        assert refusal.denominator == 1, f"{row['condition_id']}: refusal change is not a whole number of requests"
        expected[(row["target"], float(row["alpha"]))] = (refusal, Fraction(row["mmlu_paired_change_fraction"]) * 100)
    selected = [row for row in _read_rows(DOSE_SELECTION_CSV) if row["selected"] == "True"]
    assert sorted(row["target"] for row in selected) == sorted(DIRECTIONS), "one frozen dose per direction expected"
    checked = 0
    for index in (0, 1):
        for direction in DOMAINS:
            line = artists[f"development-{index}-{direction}"]
            doses = [dose for target, dose in sorted(expected) if target == direction]
            assert list(line.get_xdata()) == [_dodged(direction, dose) for dose in doses], f"{direction}: x positions"
            for dose, drawn in zip(doses, line.get_ydata()):
                assert abs(drawn - float(expected[(direction, dose)][index])) < 1e-9, f"{direction} {dose}: {drawn}"
                checked += 1
        for row in selected:
            direction, dose = row["target"], float(row["alpha"])
            marker = artists[f"development-{index}-{direction}-frozen"]
            assert list(marker.get_xdata()) == [_dodged(direction, dose)], f"{direction}: frozen dose position"
            assert abs(marker.get_ydata()[0] - float(expected[(direction, dose)][index])) < 1e-9
            checked += 1
    return checked


# ---------------------------------------------------------------- Figure 4.2: final contrasts

_COMPARISON = re.compile(r"^final-(?P<target>[a-z_]+)-real__minus__final-(?P<other>[a-z_0-9-]+)$")


def _comparator(target: str, other: str) -> str | None:
    if other == "baseline":
        return "baseline"
    for seed in ("42", "43"):
        if other == f"{target}-random{seed}":
            return f"R{seed}"
    return None  # domain − pooled contrasts are descriptive and not plotted


def _final_data() -> dict[tuple[str, str, str], tuple[float, float, float]]:
    """Return {(endpoint, direction, comparator): (estimate, lower, upper)} in percentage points."""
    data = {}
    for row in _read_rows(FINAL_PLOT_CSV):
        match = _COMPARISON.match(row["comparison_id"])
        comparator = _comparator(match["target"], match["other"]) if match else None
        if comparator is None or row["family"] not in ("real_minus_baseline", "real_minus_random"):
            continue
        assert row["unit"] == "proportion_difference" and row["interval_status"] == "nominal", row["comparison_id"]
        key = (row["endpoint"], match["target"], comparator)
        data[key] = tuple(float(row[field]) * 100 for field in ("estimate", "ci_low", "ci_high"))
    return data


def draw_final() -> plt.Figure:
    data = _final_data()
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_CM * CM, 8.4 * CM), sharey=True, layout="constrained")
    rows = {direction: len(DIRECTIONS) - 1 - position for position, direction in enumerate(DIRECTIONS)}
    panels = (
        (axes[0], "harmless", "(a) Refusal proxy, 245 groups", (-3, 60), 10),
        (axes[1], "mmlu", "(b) MMLU accuracy, 1,710 questions", (-3.2, 1.2), 1),
    )
    for ax, endpoint, title, limits, step in panels:
        ax.set_title(title.replace(", ", ",\n"), fontsize=9, color=INK)
        ax.set_xlim(*limits)
        ax.set_xticks(range(int(limits[0] // step * step + step), int(limits[1]) + 1, step))
        ax.set_xlabel("Change (percentage points)")
        ax.grid(axis="x", color=GRID, linewidth=0.5)
        ax.set_axisbelow(True)
        for boundary in (0.5, 1.5, 2.5):
            ax.axhline(boundary, color=GRID, linewidth=0.5, zorder=0)
        ax.axvline(0, color=INK_SECONDARY, linewidth=0.8, zorder=1)
        ax.tick_params(axis="y", length=0)
        for direction in DIRECTIONS:
            for comparator in COMPARATORS:
                estimate, lower, upper = data[(endpoint, direction, comparator)]
                y = rows[direction] + COMPARATOR_OFFSET[comparator]
                gid = f"final-{endpoint}-{direction}-{comparator}"
                ax.hlines(y, lower, upper, color=COLOUR[direction], linewidth=1.3, zorder=2, gid=f"{gid}-interval")
                ax.plot(
                    [estimate],
                    [y],
                    linestyle="none",
                    marker=COMPARATOR_MARKER[comparator],
                    markersize=5.5,
                    markerfacecolor=COLOUR[direction],
                    markeredgecolor="white",
                    markeredgewidth=0.7,
                    zorder=3,
                    gid=f"{gid}-estimate",
                )
    axes[0].set_yticks([rows[direction] for direction in DIRECTIONS], [LABEL[d] for d in DIRECTIONS])
    axes[0].set_ylim(-0.55, len(DIRECTIONS) - 0.45)
    axes[0].tick_params(axis="y", labelcolor=INK)
    handles = [
        Line2D(
            [],
            [],
            color=MUTED,
            linewidth=1.3,
            marker=COMPARATOR_MARKER[comparator],
            markersize=5.5,
            markerfacecolor=MUTED,
            markeredgecolor="white",
            markeredgewidth=0.7,
            label=COMPARATOR_LABEL[comparator],
        )
        for comparator in COMPARATORS
    ]
    fig.legend(handles=handles, loc="outside upper center", ncol=3, frameon=False, columnspacing=2.0)
    return fig


def check_final(fig: plt.Figure) -> int:
    """Compare drawn estimates and interval ends with the CSV ."""
    artists = {artist.get_gid(): artist for artist in fig.findobj() if artist.get_gid()}
    checked = 0
    expected = {}
    for row in _read_rows(FINAL_PLOT_CSV):
        target, _, other = row["comparison_id"].removeprefix("final-").partition("-real__minus__final-")
        comparator = {"baseline": "baseline", f"{target}-random42": "R42", f"{target}-random43": "R43"}.get(other)
        if comparator is None:
            assert row["family"] == "domain_minus_pooled_descriptive", row["comparison_id"]
            continue
        values = tuple(float(row[field]) * 100 for field in ("estimate", "ci_low", "ci_high"))
        expected[(row["endpoint"], target, comparator)] = values
        gid = f"final-{row['endpoint']}-{target}-{comparator}"
        marker, interval = artists[f"{gid}-estimate"], artists[f"{gid}-interval"]
        assert abs(marker.get_xdata()[0] - values[0]) < 1e-9, f"{gid}: estimate"
        ((lower, y_lower), (upper, y_upper)) = interval.get_segments()[0]
        assert abs(lower - values[1]) < 1e-9 and abs(upper - values[2]) < 1e-9, f"{gid}: interval"
        assert y_lower == y_upper == marker.get_ydata()[0], f"{gid}: interval and estimate rows differ"
        checked += 3
    assert len(expected) == len(DIRECTIONS) * len(COMPARATORS) * 2, f"{len(expected)} plotted contrasts"
    return checked


# ---------------------------------------------------------------- Figure 3.1: pipeline schematic

# Fixed methods schematic from the accepted figure source.
STAGES = (('Extraction sets',
  ((('refusal',
     'Harmful requests: pooled 28; cyber intrusion, dangerous substances and disinformation 24 '
     'each (six domain requests also in the pooled set). One harmless mean of 32 requests, shared '
     'by all four.',
     ()),
    ('ab',
     'Honesty: a TruthfulQA question with a supported and a contradicted answer. Sycophancy: a '
     "user's stated view on an NLP claim. Each item in both answer orders.",
     ())),)),
 ('Directions',
  ((('refusal',
     'Difference in means of the block-14 output at the final prompt token, harmful minus '
     'harmless: pooled and three domain directions; raw norms 10.91–12.34, not equalised.',
     ()),
    ('ab',
     'Contrastive Activation Addition: block-18 output at the source-labelled answer letter minus '
     'the other letter, averaged over both orders and all items; scaled to norm 12.3354.',
     ())),)),
 ('Development and dose selection',
  ((('refusal',
     '63 harmless requests; 285 MMLU questions; no random controls. Pooled fixed at α = +1; each '
     'domain takes the α in {+0.5, +1, +2} minimising '
     '$|\\Delta_\\mathrm{domain}(\\alpha)-\\Delta_\\mathrm{pooled}(+1)|$, Δ being the change in '
     'refusal-proxy rate from baseline. No MMLU or final outcome entered the rule.',
     ()),
    ('ab',
     'Honesty 64 fact groups; sycophancy 7 question groups. For each sign, the α in {±0.5, ±1, ±2} '
     'with the largest change from baseline in that direction.',
     ())),)),
 ('Frozen doses',
  ((('refusal', 'Frozen before the first final condition.', ()),
    ('ab', 'Frozen on 2 October 2026.', ())),)),
 ('Held-out final panels',
  ((('both',
     'Scored once at the frozen doses: each real direction, the unsteered baseline, and R42 and '
     "R43, fixed standard normal vectors (seeds 42 and 43) rescaled to the real direction's norm.",
     ()),),
   (('refusal',
     '246 harmless requests in 245 groups: refusal proxy (any of 12 fixed phrases). 63-request '
     'quality subset: KL and NLL. 1,710 MMLU questions: accuracy.',
     ()),
    ('ab',
     'Honesty 128 fact groups; sycophancy 8 question groups: conditional preference '
     '$p_\\mathrm{label}/(p_A+p_B)$, averaged over option orders.',
     ())))),
 ('Statistics',
  ((('both',
     'Paired contrasts with nominal two-sided 95% BCa bootstrap intervals. Refusal proxy: support '
     'rule of Table\xa03.2; against random directions, real\xa0−\xa0R42 and real\xa0−\xa0R43 must '
     'each qualify; domain\xa0−\xa0pooled descriptive. A/B: real\xa0−\xa0baseline primary; real\xa0'
     '−\xa0R42 and real\xa0−\xa0R43 must both exclude zero in the expected direction. MMLU: '
     'estimation only.',
     ()),),)),
 ('Audit',
  ((('refusal',
     'Audit of 56 final harmless responses: procedurally blinded labels, then a later informed '
     'adjudication by the author; checks the proxy and response quality.',
     ()),),)))
HEADERS = {'refusal': ('Refusal comparison · block 14', ()),
 'ab': ('A/B: honesty and sycophancy · block 18', ())}
# Horizontal extents in centimetres; "both" spans the two track columns.
COLUMNS = {"stage": (0.0, 1.95), "refusal": (2.05, 8.95), "ab": (9.1, 15.97), "both": (2.05, 15.97)}
BOX_FONT_PT, STAGE_FONT_PT, HEADER_FONT_PT = 8.0, 8.0, 8.5
PAD_CM, ARROW_CM, SUBROW_GAP_CM, LINESPACING = 0.1, 0.32, 0.15, 1.2


def _text_height_cm(fig: plt.Figure, text: str, size: float, weight: str = "normal") -> float:
    artist = Text(0, 0, text, fontsize=size, fontweight=weight, linespacing=LINESPACING, figure=fig)
    return artist.get_window_extent(fig.canvas.get_renderer()).height / fig.dpi / CM


def _line_width_cm(line: str, size: float, weight: str = "normal") -> float:
    """Width of one line as the SVG backend draws it: unhinted glyph outlines (TextToPath), in cm.

    The Agg renderer's hinted metrics are narrower, so wrapping by them let SVG text cross box edges.
    """
    font = FontProperties(family=plt.rcParams["font.family"], size=size, weight=weight)
    width_pt, _, _ = TextToPath().get_text_width_height_descent(line, font, ismath=line.count("$") >= 2)
    # PNG uses hinted Agg glyphs; take the larger width so both formats fit.
    dpi = plt.rcParams["figure.dpi"]
    agg_width, _, _ = RendererAgg(1, 1, dpi).get_text_width_height_descent(
        line, font, ismath=line.count("$") >= 2
    )
    return max(width_pt / 72, agg_width / dpi) * 2.54


def _wrap(text: str, width_cm: float, size: float, weight: str = "normal") -> str:
    """Greedy word wrap by the SVG text metrics.

    Splits at ASCII spaces only, so a no-break space keeps a reference such as "Table 3.2" whole;
    math spans are written without spaces for the same reason.
    """
    lines: list[str] = []
    for word in text.split(" "):
        candidate = f"{lines[-1]} {word}" if lines else word
        if lines and _line_width_cm(candidate, size, weight) <= width_cm:
            lines[-1] = candidate
        else:
            lines.append(word)
    return "\n".join(lines)


def draw_pipeline() -> tuple[plt.Figure, list[tuple[Text, FancyBboxPatch]]]:
    fig = plt.figure(figsize=(WIDTH_CM * CM, 20 * CM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    inner = {name: (right - left) - 2 * PAD_CM for name, (left, right) in COLUMNS.items()}

    # Lay out top to bottom in centimetres; y grows downwards until the axis is inverted.
    y = 0.05
    header_height = max(_text_height_cm(fig, title, HEADER_FONT_PT, "bold") for title, _ in HEADERS.values())
    for column, (title, _) in HEADERS.items():
        left, right = COLUMNS[column]
        ax.text((left + right) / 2, y, title, ha="center", va="top", fontsize=HEADER_FONT_PT, fontweight="bold", color=INK)
    y += header_height + 0.18
    placed: list[tuple[Text, FancyBboxPatch]] = []
    # Track -> bottom y of the previous stage's last box in that track; arrows join stages only,
    # so the sub-rows of one stage read as a group.
    previous: dict[str, float] = {}
    for stage, subrows in STAGES:
        stage_top = y
        for position, subrow in enumerate(subrows):
            wrapped = [(column, _wrap(text, inner[column], BOX_FONT_PT)) for column, text, _ in subrow]
            height = max(_text_height_cm(fig, text, BOX_FONT_PT) for _, text in wrapped) + 2 * PAD_CM
            current = {}
            for column, text in wrapped:
                left, right = COLUMNS[column]
                box = FancyBboxPatch(
                    (left, y),
                    right - left,
                    height,
                    boxstyle="round,pad=0,rounding_size=0.1",
                    facecolor=BOX_FILL,
                    edgecolor=BOX_EDGE,
                    linewidth=0.6,
                )
                ax.add_patch(box)
                label = ax.text(left + PAD_CM, y + PAD_CM, text, ha="left", va="top", fontsize=BOX_FONT_PT,
                                linespacing=LINESPACING, color=INK)
                placed.append((label, box))
                tracks = ("refusal", "ab") if column == "both" else (column,)
                for track in tracks:
                    centre = sum(COLUMNS[track]) / 2
                    if position == 0 and track in previous:
                        ax.add_patch(
                            FancyArrowPatch(
                                (centre, previous[track]),
                                (centre, y),
                                arrowstyle="-|>",
                                mutation_scale=7,
                                color=INK_SECONDARY,
                                linewidth=0.8,
                                shrinkA=0,
                                shrinkB=0,
                            )
                        )
                    current[track] = y + height
            y += height + SUBROW_GAP_CM
        previous = current
        y -= SUBROW_GAP_CM
        stage_text = _wrap(stage, COLUMNS["stage"][1] - COLUMNS["stage"][0], STAGE_FONT_PT, "bold")
        ax.text(
            COLUMNS["stage"][0],
            (stage_top + y) / 2,
            stage_text,
            ha="left",
            va="center",
            fontsize=STAGE_FONT_PT,
            fontweight="bold",
            linespacing=LINESPACING,
            color=INK,
        )
        y += ARROW_CM
    height_cm = y - ARROW_CM + 0.05
    fig.set_size_inches(WIDTH_CM * CM, height_cm * CM)
    ax.set_xlim(0, WIDTH_CM)
    ax.set_ylim(height_cm, 0)
    return fig, placed


def check_pipeline(fig: plt.Figure, placed: list[tuple[Text, FancyBboxPatch]]) -> int:
    """Check that every schematic label stays inside its box."""
    checked = 0
    fig.canvas.draw()
    for text, box in placed:
        # Widths by the SVG metrics, which the PDF shows; heights from the drawn extents.
        widest = max(_line_width_cm(line, BOX_FONT_PT) for line in text.get_text().split("\n"))
        inner, outer = text.get_window_extent(), box.get_window_extent()
        assert widest <= box.get_width() - 2 * PAD_CM, f"Figure 3.1 text overflows its box: {text.get_text()[:60]!r}"
        assert inner.y0 >= outer.y0 and inner.y1 <= outer.y1, f"Figure 3.1 text overflows vertically: {text.get_text()[:60]!r}"
        checked += 1
    return checked


# ---------------------------------------------------------------- shared


def main() -> None:
    global OUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    OUT_DIR = args.output_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    jobs = (
        ("figure-3-1-pipeline", draw_pipeline, check_pipeline),
        ("figure-4-1-development", draw_development, check_development),
        ("figure-4-2-final-contrasts", draw_final, check_final),
    )
    for stem, draw, check in jobs:
        drawn = draw()
        fig, extra = (drawn, ()) if isinstance(drawn, plt.Figure) else (drawn[0], (drawn[1],))
        _check_fonts_and_width(fig, stem)
        checked = check(fig, *extra)
        paths = _save(fig, stem)
        plt.close(fig)
        height_cm = fig.get_figheight() / CM
        print(f"{stem}: {checked} checks passed; {WIDTH_CM:.1f} x {height_cm:.2f} cm")
        for path in paths:
            print(f"  {hashlib.sha256(path.read_bytes()).hexdigest()}  {path}")
    for path, digest in SOURCE_SHA256.items():
        print(f"source verified: {digest}  {path}")


if __name__ == "__main__":
    main()
