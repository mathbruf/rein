"""Shared visual identity for the public analysis charts (output/analysis/).

These charts are the ONLY project artifacts published on GitHub (publicity
decision 2026-09-04): no raw data, no maps — the charts must therefore stand
alone: every figure carries the project identity, the study area, the data
credits and the generation date.

Palette: three hues (#eb6834 / #2a78d6 / #1b9e77), CVD-validated with the
dataviz six-checks against the light surface #fcfcfb. Identity is fixed and
never re-ordered across charts:
    C_RAW  (orange) — whole-field / effort-confounded view; insect drive
    C_FAIR (blue)   — effort-matched view; shelter drive
    C_KIND (teal)   — position-confident tier (the headline)
"""
from __future__ import annotations

import datetime as _dt

C_RAW = "#eb6834"
C_FAIR = "#2a78d6"
C_KIND = "#1b9e77"
INK, INK2, GRIDC, SURF = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"

RC = {"font.size": 11, "axes.edgecolor": INK2, "text.color": INK,
      "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
      "figure.facecolor": SURF, "axes.facecolor": SURF,
      "savefig.facecolor": SURF}

CREDITS = ("Weather: Open-Meteo (MET Nordic 1 km / ERA5, CC-BY 4.0) · "
           "Terrain: Kartverket 50 m DTM · Land cover: NIBIO AR50 · "
           "Infrastructure: OSM + Lesja fjellstyre · "
           "Field reports: villreinutvalet.no (validation only)")


def brand_footer(fig, extra: str | None = None) -> None:
    """One consistent identity strip at the very bottom of a public chart."""
    left = ("Villrein movement analysis — Lordalen, Reinheimen villreinområde "
            "(Lesja, Norway) · rule-based, weights never fitted to sightings")
    right = f"generated {_dt.date.today().isoformat()}"
    y = 0.012
    fig.text(0.008, y, left + (f" · {extra}" if extra else ""),
             fontsize=7.5, color=INK2, ha="left", va="bottom")
    fig.text(0.992, y, right, fontsize=7.5, color=INK2, ha="right", va="bottom")
    fig.text(0.008, y + 0.028, CREDITS, fontsize=7.5, color=INK2,
             ha="left", va="bottom")


def despine(ax, axis: str = "y") -> None:
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(True, axis=axis, color=GRIDC, lw=0.8)
    ax.set_axisbelow(True)
