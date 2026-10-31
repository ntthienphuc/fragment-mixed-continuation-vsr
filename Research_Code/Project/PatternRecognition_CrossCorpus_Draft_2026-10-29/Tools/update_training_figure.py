"""Render the distinct original and matched-crop comparisons as a compact vector figure."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
figure, axes = plt.subplots(figsize=(10, 5.8))
axes.set_xlim(0, 10)
axes.set_ylim(0, 5.8)
axes.axis("off")


def box(left, bottom, width, height, title, body, color):
    patch = FancyBboxPatch((left, bottom), width, height,
                          boxstyle="round,pad=0.02,rounding_size=0.07",
                          linewidth=0.8, edgecolor=color, facecolor="#f8fafc")
    axes.add_patch(patch)
    axes.text(left + 0.13, bottom + height - 0.18, title, fontsize=12.4,
              fontweight="bold", color=color, va="top")
    axes.text(left + 0.13, bottom + height - 0.48, body, fontsize=10.7,
              color="#253247", va="top", linespacing=1.3)


axes.text(0.15, 5.5, "(a) Original training comparison", fontsize=14,
          fontweight="bold", color="#193b5b")
axes.text(5.23, 5.5, "(b) Matched crop comparison", fontsize=14,
          fontweight="bold", color="#193b5b")
axes.text(5.23, 5.15, "New states; shared fallback positions", fontsize=10.7, color="#526073")

box(0.2, 4.1, 4.55, 1.0, "FC  |  Full clips", "Complete input and complete transcript", "#285782")
box(0.2, 3.0, 4.55, 1.0, "FM  |  Full + fixed-time fragments",
    "Alternate complete clips and 2 s crops\nCrop labels include overlapping words", "#19786e")
box(0.2, 1.9, 4.55, 1.0, "FM-ST  |  Full-view self-teaching",
    "FM inputs + detached full-view predictions\nTests predictive context", "#77508c")
box(0.2, 0.8, 4.55, 1.0, "CP  |  Context-preserving fragments",
    "Restore neighboring frames; keep crop labels\nTests visible context", "#a26322")

box(5.25, 3.88, 4.55, 1.1, "Full-only", "Complete input and complete transcript", "#285782")
box(5.25, 2.55, 4.55, 1.1, "Fixed-time mixing",
    "Full clips + 2 s crops\nOverlap-based labels", "#19786e")
box(5.25, 1.22, 4.55, 1.1, "Whole-word mixing",
    "Full clips + 2.00–3.28 s word-aligned spans\nConsecutive complete word annotations", "#b45d33")
axes.text(7.52, 0.88, "Same source schedule, seed and update count",
          fontsize=10.5, color="#526073", ha="center")

axes.plot([0.2, 9.8], [0.61, 0.61], color="#b8c4d0", linewidth=1)
axes.text(5, 0.32, "All states: complete video → unchanged recognizer → complete transcript",
          fontsize=12, fontweight="bold", color="#193b5b", ha="center", va="center")
figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
figure.savefig(ROOT / "Source/figures/training_design.pdf", bbox_inches="tight", pad_inches=0.03)
figure.savefig(ROOT / "Review_06_External/Training_Design.png", dpi=170,
               bbox_inches="tight", pad_inches=0.03)
plt.close(figure)
