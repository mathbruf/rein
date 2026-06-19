"""Phase 3: rule-based daily scoring engine (v0).

Turns next-day weather + the static terrain layers into a 0..1 presence score per
250 m cell. This is the heart of the project: a transparent, expert-tunable scoring
function — NOT a fitted model. Every weight and threshold below is a named constant
with a comment so a domain expert (the hunter) can tune it. Sightings are not used
here; they are kept for Phase-5 validation only.

Behavioral logic (CLAUDE.md §1), encoded as two competing weather-gated regimes:

  BASELINE: a gentle always-on preference for high ground (hunter: "mostly 1300 m+"),
     which the two weather regimes (and later, hunting disturbance) pull them off.

  INSECT / THERMAL regime  (warm + calm + DRY days):
     mosquitoes / warble & bot flies push reindeer UPHILL to wind-exposed, cool
     ground -> favor high elevation + positive TPI (ridges/summits/exposed).
     Rain grounds the flies, so this drive switches off when it rains.

  SHELTER regime  (cold OR wet OR very windy days):
     animals seek shelter and graze lower / leeward -> favor lower elevation +
     negative TPI (valleys/hollows). Per the hunter, weather/shelter dominates the
     day-to-day movement more than insects do (W_SHELTER > W_INSECT).

  FORAGE: NIBIO AR50 land cover -> a per-cell destination value (open alpine ground
     and mire score high; forest lower; water/glacier/built ~0). Additive (W_FORAGE).

  TRAVEL-LIMIT penalty: very steep/cliffed cells are down-weighted (hard to use).

  DISTURBANCE penalty: cells near roads/trails/cabins are discounted (W_DISTURB),
     fading with distance — "they come lower only if hunters allow".

The two regimes are gated by weather "pressures" in [0,1], so a warm calm day makes
the high ground light up and a cold windy day flips the surface to the valleys — the
project's core thesis, visible in one function.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- terrain normalisation anchors (fixed, so scores are reproducible & comparable
#     across days rather than relative to whatever cells were loaded) ---------------
ELEV_LO_M = 1000.0    # below here all ~equally "low" -> elev_norm 0  [hunter: mostly 1300 m+]
ELEV_HI_M = 1900.0    # ~high open fjell             -> elev_norm 1
TPI_SCALE_M = 60.0    # TPI mapped from [-60,+60] m onto [0,1]; >0 = exposed high
SLOPE_STEEP_LO = 30.0  # slope (deg) where the travel penalty starts
SLOPE_STEEP_HI = 45.0  # slope (deg) at/above which the penalty is full
DISTURB_DECAY_M = 2500.0  # disturbance penalty fades from full (at a feature) to 0 by this distance

# --- weather -> pressure ramps ---------------------------------------------------
# Insect activity climbs with warmth and is suppressed by wind.
INSECT_T_LO, INSECT_T_HI = 10.0, 18.0   # degC: <10 ~no insects, >18 ~full pressure.
                                        # ASSUMPTION (hunter unsure 2026-06-19) - test in validation.
INSECT_W_CALM, INSECT_W_BREEZY = 2.0, 5.0  # m/s: full bug pressure when calm (<=2), gone by a
                                           # light breeze (~5) [hunter, 2026-06-19]
INSECT_RAIN_OFF_MM = 2.0  # rain grounds the flies: bug drive ~0 by ~2 mm/day [hunter: no bugs in rain]
# Shelter drivers.
COLD_T_HI, COLD_T_LO = 8.0, 0.0    # degC: >=8 ~no cold drive, <=0 ~full. ASSUMPTION (hunter no info)
WET_MM_FULL = 5.0                  # mm/day giving full "wet" drive.     ASSUMPTION (hunter no info)
WIND_HI_LO, WIND_HI_HI = 8.0, 15.0  # m/s: strong-wind shelter drive ramp

# --- regime / term weights (expert-set; tune these) ------------------------------
# Hunter 2026-06-19: weather/shelter dominates day-to-day over insects -> shelter > insect.
W_INSECT = 0.7     # weight of the insect/thermal (go-high) regime
W_SHELTER = 1.0    # weight of the shelter (go-low) regime
W_STEEP = 0.5      # strength of the steep-terrain travel penalty (0..1 multiplier)
W_BASELINE = 0.3   # gentle always-on pull to high ground [hunter: "mostly 1300 m+"]; weather and
                   #   (later) disturbance override it - "lower if wind and hunters allow"
W_DISTURB = 0.6    # disturbance penalty strength [hunter: "lower only if hunters allow"]
W_FORAGE = 0.4     # additive forage destination value (NIBIO AR50 land cover)

# within-regime terrain mix (how much elevation vs exposure each regime cares about)
REFUGE_ELEV_W, REFUGE_TPI_W = 0.6, 0.4   # insect/thermal: high + exposed
SHELTER_ELEV_W, SHELTER_TPI_W = 0.6, 0.4  # shelter: low + sheltered


@dataclass
class WeatherDay:
    """Area-wide next-day forecast scalars (v0 uses one value for the whole field;
    spatial weather interpolation is a Phase-4 refinement)."""
    temp_c: float
    wind_ms: float
    precip_mm: float


def _ramp(x, lo, hi):
    """Linear 0->1 as x goes lo->hi (clipped). Handle hi<lo (descending) too."""
    if hi == lo:
        return np.where(x >= hi, 1.0, 0.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def insect_pressure(w: WeatherDay) -> float:
    warm = float(_ramp(w.temp_c, INSECT_T_LO, INSECT_T_HI))
    calm = float(_ramp(w.wind_ms, INSECT_W_BREEZY, INSECT_W_CALM))  # descending
    dry = 1.0 - float(_ramp(w.precip_mm, 0.0, INSECT_RAIN_OFF_MM))  # rain grounds the flies
    return warm * calm * dry


def shelter_pressure(w: WeatherDay) -> float:
    cold = float(_ramp(w.temp_c, COLD_T_HI, COLD_T_LO))   # descending
    wet = float(_ramp(w.precip_mm, 0.0, WET_MM_FULL))
    windy = float(_ramp(w.wind_ms, WIND_HI_LO, WIND_HI_HI))
    # probabilistic OR: any driver can push the animals to shelter
    return 1.0 - (1.0 - cold) * (1.0 - wet) * (1.0 - windy)


def _terrain_norm(elev, tpi):
    elev_n = np.clip((elev - ELEV_LO_M) / (ELEV_HI_M - ELEV_LO_M), 0.0, 1.0)
    tpi_n = np.clip((tpi + TPI_SCALE_M) / (2 * TPI_SCALE_M), 0.0, 1.0)  # 0=hollow,1=ridge
    return elev_n, tpi_n


def score_cells(elev, slope, tpi, w: WeatherDay,
                disturb_dist=None, forage=None) -> dict[str, np.ndarray]:
    """Score arrays of per-cell terrain attributes for one weather day.

    disturb_dist: optional per-cell distance (m) to nearest disturbance feature;
    when given (and W_DISTURB>0) cells near roads/trails/cabins are penalised.
    forage: optional per-cell forage value 0..1 (AR50); added when W_FORAGE>0.

    Returns raw score plus the regime pressures used (for explanation) and a
    0..1 normalised score (min-max over the scored cells) for the heatmap.
    """
    elev = np.asarray(elev, float)
    slope = np.asarray(slope, float)
    tpi = np.asarray(tpi, float)
    elev_n, tpi_n = _terrain_norm(elev, tpi)

    refuge = REFUGE_ELEV_W * elev_n + REFUGE_TPI_W * tpi_n            # go-high target
    shelter = SHELTER_ELEV_W * (1 - elev_n) + SHELTER_TPI_W * (1 - tpi_n)  # go-low target

    p_ins = insect_pressure(w)
    p_shl = shelter_pressure(w)

    baseline = elev_n   # default home-range preference: high ground (hunter: "mostly 1300 m+")
    base = ((W_BASELINE * baseline)
            + (W_INSECT * p_ins * refuge)
            + (W_SHELTER * p_shl * shelter))
    if forage is not None and W_FORAGE > 0:
        base = base + W_FORAGE * np.asarray(forage, float)  # forage destination value
    steep = _ramp(slope, SLOPE_STEEP_LO, SLOPE_STEEP_HI)
    raw = base * (1.0 - W_STEEP * steep)
    if disturb_dist is not None and W_DISTURB > 0:
        # near a road/trail/cabin -> penalty 1, fading to 0 by DISTURB_DECAY_M
        disturb_pen = _ramp(np.asarray(disturb_dist, float), DISTURB_DECAY_M, 0.0)
        raw = raw * (1.0 - W_DISTURB * disturb_pen)

    lo, hi = np.nanmin(raw), np.nanmax(raw)
    norm = (raw - lo) / (hi - lo) if hi > lo else np.zeros_like(raw)
    return {"score_raw": raw, "score": norm,
            "insect_pressure": np.full(raw.shape, p_ins),
            "shelter_pressure": np.full(raw.shape, p_shl)}
