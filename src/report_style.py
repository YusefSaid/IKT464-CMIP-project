"""Figure style for the report: figures drawn at the size they are printed.

Principle
    LaTeX scales a figure to the width it is included at. A figure drawn 20 inches wide
    and shown at the 16.5 cm text width shrinks its text to about 4 pt. Drawing it at
    the printed size (TEXTWIDTH) keeps the 9 pt text 9 pt, close to the 11 pt body text.
    Include the PDFs with width=\\textwidth (or the fraction used in figsize).

Usage in a notebook:
    from report_style import TEXTWIDTH, use_report_style, map_panel, hatch, colorbar
    use_report_style()
    fig, axs = plt.subplots(1, 2, figsize=(TEXTWIDTH, 2.4), layout="constrained")
    m = map_panel(axs[0], grid, values, cmap="RdBu_r", vmin=-0.4, vmax=0.4, title="...")
    colorbar(fig, m, axs, "Δ Hs [m]")
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from cmip_utils import to_map

TEXTWIDTH = 6.5          # inches = 165 mm, \textwidth of the report
FONT = 9                 # pt; report body text is 11 pt


def use_report_style():
    plt.rcParams.update({
        "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
        "xtick.labelsize": FONT - 1, "ytick.labelsize": FONT - 1,
        "legend.fontsize": FONT - 1, "figure.titlesize": FONT + 1,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5, "lines.linewidth": 1.2,
        "hatch.color": "0.15", "hatch.linewidth": 0.35,
        "figure.dpi": 130,          # on-screen size in the notebook
        "savefig.dpi": 300, "savefig.pad_inches": 0.02,
        "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
    })


LON_TICKS = [0, 90, 180, 270, 360]
LON_LABELS = ["0°", "90°E", "180°", "90°W", "0°"]
LAT_TICKS = [-60, -30, 0, 30, 60]
LAT_LABELS = ["60°S", "30°S", "0°", "30°N", "60°N"]


def map_panel(ax, grid, values, cmap="viridis", vmin=None, vmax=None, title="",
              left=True, bottom=True):
    """Draw one global map (one value per sea point) without a colour bar.

    Land and masked points are light grey. Tick labels only on the left column
    (left=True) and bottom row (bottom=True). Returns the mesh for colorbar().
    """
    m = ax.pcolormesh(grid["lon"], grid["lat"], to_map(grid, values), shading="auto",
                      cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_facecolor("lightgrey")
    ax.set_xlim(0, 360)
    ax.set_ylim(-80, 90)
    ax.set_aspect("equal")
    ax.set_xticks(LON_TICKS, LON_LABELS if bottom else [])
    ax.set_yticks(LAT_TICKS, LAT_LABELS if left else [])
    if title:
        ax.set_title(title, pad=3)
    return m


def hatch(ax, grid, mask):
    """Dot the points where mask is True (e.g. significant change)."""
    if np.any(mask):
        ax.contourf(grid["lon"], grid["lat"], to_map(grid, np.asarray(mask, float)),
                    levels=[0.5, 1.5], colors="none", hatches=["...."])


def colorbar(fig, mesh, axs, label, orientation="horizontal", **kw):
    """One colour bar for one or several panels (below them by default)."""
    opts = dict(shrink=0.6, aspect=35, pad=0.02) if orientation == "horizontal" \
        else dict(shrink=0.9, aspect=18, pad=0.02)
    opts.update(kw)
    return fig.colorbar(mesh, ax=axs, orientation=orientation, label=label, **opts)


def symmetric_limit(*arrays, q=99):
    """Colour-scale limit for differences: q-th percentile of |values| over all arrays."""
    return float(np.nanpercentile(np.abs(np.concatenate([np.ravel(a) for a in arrays])), q))
