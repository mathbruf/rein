"""Phase 3: rule-based daily scoring engine (v1 — per-cell weather).

Turns next-day weather + the static terrain layers into a 0..1 presence score per
250 m cell. This is the heart of the project: a transparent, expert-tunable scoring
function — NOT a fitted model. Every weight and threshold below is a named constant
with a comment so a domain expert (the field expert) can tune it. Sightings are not used
here; they are kept for Phase-5 validation only.

Behavioral logic (CLAUDE.md §1), encoded as two competing weather-gated regimes over
a *per-cell downscaling* of the area forecast, so the map is never flat on a calm day:

  WEATHER DOWNSCALING (new in v1; physics, not fitted):
     one area forecast (a single temp/wind for the field) is turned into a per-cell
     field. Temperature falls with elevation at the standard environmental lapse rate
     (~6.5 °C/km), so the tops are modeled colder than the valley reading; wind is
     amplified on wind-exposed ground (+TPI ridges/summits) and damped in hollows
     (−TPI). This is the single biggest reason the old area-wide scorer went
     *uninformative* for the autumn observation window: on a "calm, cool" autumn day the valley
     reading turned BOTH regimes off, yet the exposed tops are genuinely cold and
     windy. Downscaling lets the shelter regime engage from the terrain itself.

  DAY DRIVERS (region-level regime switches, from the downscaled field):
     insect driver  = insect pressure in the WARM lowland (warm tail of the field):
        if flies are biting anywhere it is in the low, calm, sunny ground.
     shelter driver = harshness on the EXPOSED tops (cold/windy tail of the field):
        if it is cold/windy anywhere the exposed animals feel it first.

  INSECT / THERMAL regime  (warm + calm + DRY day):
     mosquitoes / warble & bot flies push reindeer UPHILL off the biting lowland ->
     a cell is favored when it is itself insect-free (cold and/or windy) and
     wind-exposed. Rain grounds the flies, so the driver switches off in the wet.

  SHELTER regime  (cold OR wet OR windy day):
     animals seek shelter and graze lower / leeward -> a cell is favored when it is
     itself calm and mild (not cold, not wind-scoured) and topographically leeward.
     Per the field expert, weather/shelter dominates day-to-day movement more than insects
     do (W_SHELTER > W_INSECT).

  BASELINE: a gentle always-on preference for high ground (field expert: "mostly 1300 m+"),
     which the two weather regimes (and disturbance) pull them off.

  FORAGE: NIBIO AR50 land cover -> a per-cell destination value (open alpine ground
     and mire score high; forest lower; water/glacier/built ~0). Additive (W_FORAGE).

  TRAVEL-LIMIT penalty: very steep/cliffed cells are down-weighted (hard to use).

  DISTURBANCE penalty: cells near roads/trails/cabins are discounted (W_DISTURB),
     fading with distance — "they come lower only if human activity allows".
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# --- terrain normalisation anchors (fixed, so scores are reproducible & comparable
#     across days rather than relative to whatever cells were loaded) ---------------
from reindeer.area import AREA

ELEV_LO_M = AREA.elev_lo_m  # below here all ~equally "low" -> elev_norm 0  [field expert: mostly 1300 m+]
ELEV_HI_M = AREA.elev_hi_m  # ~high open fjell             -> elev_norm 1  (config/area.json)
TPI_SCALE_M = 60.0    # TPI mapped from [-60,+60] m onto [0,1]; >0 = exposed high
SLOPE_STEEP_LO = 30.0  # slope (deg) where the travel penalty starts
SLOPE_STEEP_HI = 45.0  # slope (deg) at/above which the penalty is full
DISTURB_DECAY_M = 2500.0  # disturbance penalty fades from full (at a feature) to 0 by this distance

# --- weather downscaling (physics; NOT fitted to sightings) ----------------------
LAPSE_C_PER_M = 6.5e-3          # environmental lapse rate ~6.5 °C per 1000 m
WIND_EXPOSURE_GAIN = 1.0        # ±100% of the area wind across the exposure range
WIND_EXPOSURE_TPI_M = 60.0      # TPI (m) mapped onto the [−1,+1] exposure scale (ridge..hollow)
DAY_WARM_PCTL = 90.0            # "the warm lowland" percentile for the insect driver
DAY_CALM_PCTL = 10.0            # calmest air (hollows) for the insect driver
DAY_COLD_PCTL = 10.0            # "the exposed tops" (coldest) for the shelter driver
DAY_WINDY_PCTL = 90.0           # windiest air (ridges) for the shelter driver

# --- REAL weather field path (the reconstruction) --------------------------------
# The scorer now consumes a per-cell WeatherField (weather/field.py): real temperature,
# wind SPEED and DIRECTION, and precipitation interpolated from a live/ERA5 lattice.
# These constants govern how that real field turns into per-cell exposure and regimes.
# No fixed lapse rate is applied here — temperature already comes from the data.
EXPOSURE_SLOPE_FULL_DEG = 20.0  # slope (deg) at which aspect exposure reaches full effect (flat=neutral)
ASPECT_EXPOSURE_W = 0.5         # directional (slope-aspect vs wind) share of the single exposure channel
TPI_EXPOSURE_W = 0.5            # landform (TPI ridge/hollow) share of the single exposure channel
W_WEATHER = 1.0                 # overall strength of the (decoupled) weather-regime term

# --- weather -> pressure ramps ---------------------------------------------------
# Insect activity climbs with warmth and is suppressed by wind.
INSECT_T_LO, INSECT_T_HI = 10.0, 18.0   # degC: <10 ~no insects, >18 ~full pressure.
                                        # ASSUMPTION (field expert unsure 2026-06-19) - test in validation.
INSECT_W_CALM, INSECT_W_BREEZY = 2.0, 5.0  # m/s: full bug pressure when calm (<=2), gone by a
                                           # light breeze (~5) [field expert, 2026-06-19]
INSECT_RAIN_OFF_MM = 2.0  # rain grounds the flies: bug drive ~0 by ~2 mm/day [field expert: no bugs in rain]
# Shelter drivers.
COLD_T_HI, COLD_T_LO = 8.0, 0.0    # degC: >=8 ~no cold drive, <=0 ~full. ASSUMPTION (field expert no info)
WET_MM_FULL = 5.0                  # mm/day giving full "wet" drive.     ASSUMPTION (field expert no info)
WIND_HI_LO, WIND_HI_HI = 8.0, 15.0  # m/s: strong-wind shelter drive ramp

# --- regime / term weights (expert-set; tune these) ------------------------------
# Field expert 2026-06-19: weather/shelter dominates day-to-day over insects -> shelter > insect.
W_INSECT = 0.7     # weight of the insect/thermal (go-high) regime
W_SHELTER = 1.0    # weight of the shelter (go-low) regime
W_STEEP = 0.5      # strength of the steep-terrain travel penalty (0..1 multiplier)
W_BASELINE = 0.3   # gentle always-on pull to high ground [field expert: "mostly 1300 m+"]; weather and
                   #   (later) disturbance override it - "lower if wind and human activity allows"
W_DISTURB = 0.6    # disturbance penalty strength [field expert: "lower only if human activity allows"]
W_FORAGE = 0.4     # additive forage destination value (NIBIO AR50 land cover)
FORAGE_RELATIVE = False  # IDEA 016: subtract the field-mean forage so the ~constant
                         #   open-alpine offset (~89% of cells at 0.85) stops
                         #   compressing the weather signal's dynamic range

# within-regime cell-quality mix (how the per-cell comfort vs terrain shape combine)
REFUGE_COOL_W, REFUGE_EXPOSE_W = 0.7, 0.3   # insect escape: insect-free air + exposed ground
SHELTER_CALM_W, SHELTER_LEE_W = 0.7, 0.3    # shelter: mild/calm air + leeward hollow


@dataclass
class WeatherDay:
    """Area-wide next-day forecast scalars (one reading for the field); the scorer
    downscales these to per-cell temperature/wind by elevation and exposure."""
    temp_c: float
    wind_ms: float
    precip_mm: float


def _ramp(x, lo, hi):
    """Linear 0->1 as x goes lo->hi (clipped). Handle hi<lo (descending) too."""
    x = np.asarray(x, float)
    if hi == lo:
        return np.where(x >= hi, 1.0, 0.0)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def _insect_p(temp_c, wind_ms, precip_mm):
    """Insect pressure (0..1) — warm AND calm AND dry. Vectorised over cells."""
    warm = _ramp(temp_c, INSECT_T_LO, INSECT_T_HI)
    calm = _ramp(wind_ms, INSECT_W_BREEZY, INSECT_W_CALM)  # descending
    dry = 1.0 - _ramp(precip_mm, 0.0, INSECT_RAIN_OFF_MM)  # rain grounds the flies
    return warm * calm * dry


def _shelter_p(temp_c, wind_ms, precip_mm):
    """Shelter drive (0..1) — probabilistic OR of cold, wet, windy. Vectorised."""
    cold = _ramp(temp_c, COLD_T_HI, COLD_T_LO)   # descending
    wet = _ramp(precip_mm, 0.0, WET_MM_FULL)
    windy = _ramp(wind_ms, WIND_HI_LO, WIND_HI_HI)
    return 1.0 - (1.0 - cold) * (1.0 - wet) * (1.0 - windy)


def insect_pressure(w: WeatherDay) -> float:
    """Area-wide insect pressure at the raw forecast reading (scalar; for reporting)."""
    return float(_insect_p(w.temp_c, w.wind_ms, w.precip_mm))


def shelter_pressure(w: WeatherDay) -> float:
    """Area-wide shelter drive at the raw forecast reading (scalar; for reporting)."""
    return float(_shelter_p(w.temp_c, w.wind_ms, w.precip_mm))


def downscale_weather(w: WeatherDay, elev, tpi, ref_elev=None):
    """Per-cell (temp_c, wind_ms) from the area forecast — physics, not fitted.

    temp: falls with elevation at the standard lapse rate relative to ref_elev (the
      elevation the area reading represents; default = field mean elevation).
    wind: scaled by terrain exposure — amplified on +TPI ridges, damped in −TPI
      hollows — so the tops are modeled windier than a valley reading.
    """
    elev = np.asarray(elev, float)
    tpi = np.asarray(tpi, float)
    if ref_elev is None:
        ref_elev = float(np.nanmean(elev))
    temp = w.temp_c - LAPSE_C_PER_M * (elev - ref_elev)
    expo = np.clip(tpi / WIND_EXPOSURE_TPI_M, -1.0, 1.0)   # −1 hollow .. +1 ridge
    wind = np.clip(w.wind_ms * (1.0 + WIND_EXPOSURE_GAIN * expo), 0.0, None)
    return temp, wind


def _terrain_norm(elev, tpi):
    elev_n = np.clip((elev - ELEV_LO_M) / (ELEV_HI_M - ELEV_LO_M), 0.0, 1.0)
    tpi_n = np.clip((tpi + TPI_SCALE_M) / (2 * TPI_SCALE_M), 0.0, 1.0)  # 0=hollow,1=ridge
    return elev_n, tpi_n


def exposure_channel(tpi, slope, aspect_deg, wind_dir_deg):
    """Single per-cell exposure in [−1,+1]: −1 = sheltered/leeward, +1 = exposed/windward.

    Combines TWO real signals through ONE channel (so terrain shape is not double
    counted — IDEA 015):
      - landform: TPI (ridge/summit +, hollow −);
      - directional: slope aspect vs the wind's 'from' direction — a slope facing into
        the wind is windward (+), the lee side is sheltered (−), scaled by steepness so
        flat ground is neutral (IDEA 012, the missing physical input).
    """
    tpi = np.asarray(tpi, float)
    slope = np.asarray(slope, float)
    tpi_signed = np.clip(tpi / WIND_EXPOSURE_TPI_M, -1.0, 1.0)
    slope_fac = np.clip(slope / EXPOSURE_SLOPE_FULL_DEG, 0.0, 1.0)
    windward = np.cos(np.radians(np.asarray(aspect_deg, float)
                                 - np.asarray(wind_dir_deg, float))) * slope_fac
    return np.clip(TPI_EXPOSURE_W * tpi_signed + ASPECT_EXPOSURE_W * windward, -1.0, 1.0)


def score_cells(elev, slope, tpi, w: WeatherDay = None,
                disturb_dist=None, forage=None,
                downscale=True, ref_elev=None,
                wfield=None, aspect=None) -> dict[str, np.ndarray]:
    """Score arrays of per-cell terrain attributes for one weather day.

    wfield: a per-cell WeatherField (weather/field.py) of REAL interpolated weather —
      temperature, wind speed + DIRECTION, precipitation. When given, this is the
      primary path: no fixed lapse rate (temperature is data), wind direction drives
      aspect-aware exposure, and the two regimes are decoupled so they switch instead
      of co-firing and cancelling (IDEAS 012/013/014). Needs per-cell `aspect` (deg).
    downscale: legacy fallback when wfield is None — the area forecast `w` is turned
      into per-cell temp/wind by a fixed lapse rate + TPI exposure. downscale=False is
      the oldest area-wide behaviour. Kept for the demo and back-compat.
    disturb_dist: optional per-cell distance (m) to nearest disturbance feature;
      when given (and W_DISTURB>0) cells near roads/trails/cabins are penalised.
    forage: optional per-cell forage value 0..1 (AR50); added when W_FORAGE>0.

    Returns raw score, a 0..1 normalised score (min-max) for the heatmap, and the
    day-level regime pressures used (filled per cell, for explanation/reporting).
    """
    elev = np.asarray(elev, float)
    slope = np.asarray(slope, float)
    tpi = np.asarray(tpi, float)
    elev_n, tpi_n = _terrain_norm(elev, tpi)

    if wfield is not None:
        # ---- REAL per-cell weather field (the reconstruction) -------------------
        if aspect is None:
            raise ValueError("wfield path needs per-cell `aspect` (deg)")
        temp_c = np.asarray(wfield.temp_c, float)
        wind_amb = np.asarray(wfield.wind_ms, float)       # real ambient wind per cell
        wind_dir = np.asarray(wfield.wind_dir_deg, float)
        precip = np.asarray(wfield.precip_mm, float)
        # one exposure channel from real TPI + real wind direction vs slope aspect
        expo = exposure_channel(tpi, slope, aspect, wind_dir)
        expo_n = 0.5 * (expo + 1.0)                          # 0 sheltered .. 1 exposed
        eff_wind = np.clip(wind_amb * (1.0 + WIND_EXPOSURE_GAIN * expo), 0.0, None)
        # per-cell comfort from REAL temperature + exposure-adjusted wind
        p_ins_cell = _insect_p(temp_c, eff_wind, precip)
        p_shl_cell = _shelter_p(temp_c, eff_wind, precip)
        refuge = REFUGE_COOL_W * (1.0 - p_ins_cell) + REFUGE_EXPOSE_W * expo_n
        shelter = SHELTER_CALM_W * (1.0 - p_shl_cell) + SHELTER_LEE_W * (1.0 - expo_n)
        # DAY-LEVEL drivers from the real field. Insects gate on the AMBIENT (synoptic)
        # wind — the field median, NOT a guaranteed-calm hollow — so the calm gate is
        # no longer trivially true (IDEA 013). Shelter reads the harsh tail of the field.
        area_precip = float(np.nanmean(precip))
        warm_low = float(np.nanpercentile(temp_c, DAY_WARM_PCTL))
        amb_wind = float(np.nanmedian(wind_amb))
        cold_top = float(np.nanpercentile(temp_c, DAY_COLD_PCTL))
        windy_top = float(np.nanpercentile(eff_wind, DAY_WINDY_PCTL))
        p_ins = float(_insect_p(warm_low, amb_wind, area_precip))
        p_shl = float(_shelter_p(cold_top, windy_top, area_precip))
        # DECOUPLED regime term: a weighted SWITCH between the go-high and go-low
        # targets (not two independent additive terms that partly cancel). The field expert's
        # shelter>insect preference sets the blend; `drive` is the overall weather push.
        wi, ws = W_INSECT * p_ins, W_SHELTER * p_shl
        r_ins = wi / (wi + ws + 1e-9)
        regime_target = r_ins * refuge + (1.0 - r_ins) * shelter
        drive = max(p_ins, p_shl)
        weather_term = W_WEATHER * drive * regime_target
    else:
        if w is None:
            raise ValueError("score_cells needs either wfield or a WeatherDay `w`")
        if downscale:
            temp_c, wind_ms = downscale_weather(w, elev, tpi, ref_elev)
            # regime switches from the field extremes: insects from the warm lowland,
            # shelter from the cold/windy exposed tops (any harsh corner grounds them).
            warm_lowland = float(np.nanpercentile(temp_c, DAY_WARM_PCTL))
            calm_lowland = float(np.nanpercentile(wind_ms, DAY_CALM_PCTL))
            cold_tops = float(np.nanpercentile(temp_c, DAY_COLD_PCTL))
            windy_tops = float(np.nanpercentile(wind_ms, DAY_WINDY_PCTL))
            p_ins = float(_insect_p(warm_lowland, calm_lowland, w.precip_mm))
            p_shl = float(_shelter_p(cold_tops, windy_tops, w.precip_mm))
            p_ins_cell = _insect_p(temp_c, wind_ms, w.precip_mm)
            p_shl_cell = _shelter_p(temp_c, wind_ms, w.precip_mm)
            refuge = REFUGE_COOL_W * (1.0 - p_ins_cell) + REFUGE_EXPOSE_W * tpi_n
            shelter = SHELTER_CALM_W * (1.0 - p_shl_cell) + SHELTER_LEE_W * (1.0 - tpi_n)
        else:
            p_ins = float(_insect_p(w.temp_c, w.wind_ms, w.precip_mm))
            p_shl = float(_shelter_p(w.temp_c, w.wind_ms, w.precip_mm))
            refuge = 0.6 * elev_n + 0.4 * tpi_n              # legacy go-high target
            shelter = 0.6 * (1 - elev_n) + 0.4 * (1 - tpi_n)  # legacy go-low target
        weather_term = (W_INSECT * p_ins * refuge) + (W_SHELTER * p_shl * shelter)

    baseline = elev_n   # default home-range preference: high ground (field expert: "mostly 1300 m+")
    base = (W_BASELINE * baseline) + weather_term
    if forage is not None and W_FORAGE > 0:
        fv = np.asarray(forage, float)
        if FORAGE_RELATIVE:                       # IDEA 016: discriminate, don't offset
            fv = fv - float(np.nanmean(fv))
        base = base + W_FORAGE * fv               # forage destination value
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


def explain_cell(elev, tpi, p_ins, p_shl,
                 dist_disturb=None, forage=None) -> str:
    """A short human-readable reason why a cell scored high, given the day's regime.

    Reflects the same logic the scorer uses, so the reason matches the number.
    """
    parts: list[str] = []
    if p_ins >= 0.3 and p_ins >= p_shl:                 # insect/thermal day -> go high+exposed
        if elev >= 1500:
            parts.append("high ground")
        if tpi >= 20:
            parts.append("wind-exposed ridge")
        elif tpi >= 0:
            parts.append("open ground")
        if not parts:
            parts.append("insect-escape terrain")
    elif p_shl >= 0.3:                                  # shelter day -> go low+leeward
        if tpi <= -30:
            parts.append("sheltered hollow")
        elif tpi < 0:
            parts.append("leeward ground")
        if elev <= 1300:
            parts.append("lower ground")
        if not parts:
            parts.append("shelter terrain")
    else:                                               # calm/mild -> baseline high-ground pref
        parts.append("high ground (baseline)" if elev >= 1400 else "mid-elevation")
    if dist_disturb is not None and dist_disturb >= 3000:
        parts.append("remote from access")
    if forage is not None and forage >= 0.8:
        parts.append("good forage")
    return ", ".join(parts)
