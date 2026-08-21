# Ideas Backlog

Statuses: proposed | accepted | done | rejected
Agents add ideas here as `proposed`; the human promotes them to ROADMAP.

| ID | Date | Status | Idea | Rationale | By |
|----|------|--------|------|-----------|----|
| 001 | YYYY-MM-DD | example | Include solar radiation / aspect as a heat-load proxy | May sharpen the warm-day "climb to cool exposed ground" signal | example |
| 002 | 2026-06-12 | proposed | Free-prose extractor for pre-2023 (2022) villreinutvalet posts | 2022 posts are narrative (no `Region:` lines); a sentence-level NLP pass could recover ~1 more season of validation sightings, at the cost of noisier landmark/count extraction | Claude |
| 003 | 2026-06-19 | proposed | Upgrade DTM 50 m → 10 m for terrain derivatives | TPI/ruggedness/slope at 250 m cells would be sharper from a 10 m DTM (25 vs 625 source pixels/cell); needs tiled WCS requests (one request would be ~29 M px). 50 m is adequate for v0 | Claude |
| 004 | 2026-06-19 | **done 2026-07-07** | Elevation-aware temperature in the scorer | Apply a lapse rate (~-0.6 °C/100 m) to the area forecast so high cells are modeled colder than the valley reading; sharpens the per-cell insect/thermal vs shelter regime split instead of using one area-wide temp. **Done in scorer v1** (`downscale_weather()`), extended to per-cell wind via terrain exposure; fixes the calm-autumn flat-map failure (favoured-half 51→57%). | Claude |
| 005 | 2026-06-19 | proposed | Weight disturbance by feature type | v1 disturbance treats a trunk road and a faint footpath equally (distance to nearest of any). Weight by class (trunk/secondary » track » path; manned cabin » trailhead) so busy access penalises more than a faint trail | Claude |
| 006 | 2026-06-19 | **done 2026-07-07** | Cluster the daily top-zones into named areas | daily_map.py lists top individual 250 m cells, which are often adjacent. Cluster high-score cells into a handful of distinct zones (centroid + extent + nearest gazetteer landmark) for a cleaner "go here" list. **Done** via `cluster_top_zones()` (greedy, min-separation) used by the redesigned map + daily/historical scripts. | Claude |
| 007 | 2026-06-19 | proposed | Refine forage with AR50 vegetasjonsdekke | Lordalen is 89% one arealtype (open alpine), so forage barely discriminates within the zone. The AR50 vegetasjonsdekke field (vegetated vs bare rock/scree) could separate prime lichen/grass from barren ground | Claude |
| 008 | 2026-06-20 | proposed | **Seasonal scorer profiles (autumn vs summer)** | With corrected (human-pinned) positions the summer baseline is NOT the culprit (the earlier "ablating both lifts AUC 0.32→0.60" was a mislocated-position artifact, since overturned). Still worth a season switch: insects are gone by Sept, so an autumn profile could lean the scorer on shelter + forage + terrain and away from the summer insect driver for late Aug–Sept. Re-test on held-out/CV data, not the same set | Claude |
| 009 | 2026-06-20 | **done 2026-08-03** | **Effort/accessibility correction for validation** | The disturbance penalty hurt validation because presence-only reports are effort-biased toward accessible terrain. **Done:** a target-group **effort-matched background** (`validation.effort_weights` + weighted percentile) reweights each field cell by the empirical density of the reports' own distance-to-disturbance, so "available" ground matches where reports occur (evaluation only; model untouched). Result under the CV harness: shipped-model CV AUC **0.498→0.587**, folds-beating-chance **47%→80%**, and the disturbance-off advantage shrinks (Δ +0.124→+0.066) — real signal was masked by the confound. Effort-matched is now the primary Phase-6 gate. Residual disturb-off edge (+0.066) means the correction is partial; a per-report accessibility covariate could refine it further. | Claude |
| 010 | 2026-06-20 | **done 2026-08-03** | k-fold cross-validation harness | Built as `model/cv.py` + `scripts/cv_validate.py` (repeated k-fold + select-then-evaluate); has been the Phase-6 gate for every change since | Claude |
| 011 | 2026-06-20 | proposed | MET Frost as historical-weather upgrade | Phase 5 used Open-Meteo ERA5 archive (no key) instead of MET Frost. With a free Frost client ID, compare nearest-station vs ERA5 and optionally switch, or keep ERA5 for gridded coverage | Claude |

## Phase 6 — Model correctness & hit-rate (weakness audit 2026-08-03)

Findings from a full read of `model/score.py` + `docs/validation_report.md`. The model currently ranks the held-out reports **at chance** (AUC 0.497, p≈0.51); these are the concrete reasons why, ordered by expected impact. Each is a **structural/physics** fix (or a measurement/display fix), NOT a weight re-tune against the 37 reports — so every one must be validated **out-of-sample under the CV harness (IDEA 010)** before it is trusted. Tier-1 items are the likeliest causes of the at-chance result.

> **Weather reconstruction landed 2026-08-09 (Tier-1 done).** The single-point + fixed-constant
> downscaling was replaced by a **real, spatially-varying per-cell weather field** (`weather/field.py`):
> a lattice of real Open-Meteo points (metno_nordic 1 km for forecasts, ERA5 for validation) fetched
> **with wind direction**, interpolated to every cell (wind as u/v vectors, temperature via a data-driven
> lapse fit, precip via IDW). This closed IDEAS 012 (aspect·wind exposure), 013 (regime decoupling +
> real ambient calm gate), 014 (inversions handled by the data-driven lapse), and most of 015 (single
> exposure channel). **CV gate (effort-matched, the primary bar): 0.587 → 0.599, folds-beating-chance
> 80% → 82%, effort-matched top-20% hit-rate 32% → 41%.** A modest but real out-of-sample gain, now on
> honest physics rather than fixed constants. Remaining: 016 forage offset, 017 rank display, 018 sweep
> the regime thresholds AND the new exposure-blend weights (`ASPECT_EXPOSURE_W`/`TPI_EXPOSURE_W`/
> `W_WEATHER`) under the select-then-evaluate CV (out-of-sample only, never hand-tuned to the 37 reports).

| ID | Date | Status | Idea | Rationale | By |
|----|------|--------|------|-----------|----|
| 012 | 2026-08-03 | **done 2026-08-09** | **Wind *direction* → aspect-aware exposure** | The v1 scorer used only wind *speed* and **neither source fetched direction**. **Done in the weather reconstruction:** `weather/field.py` fetches real wind direction (Open-Meteo `wind_direction_10m`, both metno_nordic forecast + ERA5 archive), interpolated to cells as u/v vectors; `terrain.aspect_deg` adds per-cell slope aspect; `score.exposure_channel()` makes exposure a function of `cos(aspect − wind_dir)` scaled by slope. Flipping the wind 180° now moves favoured leeward ground to opposite slopes (was zero effect). | Claude |
| 013 | 2026-08-03 | **done 2026-08-09** | **Decouple the two regimes so they *switch*, not co-fire** | **Done:** the field path replaces the two independent additive regime terms with a single **weighted-switch** term — `regime_target = r·refuge + (1−r)·shelter`, `r = Wi·p_ins/(Wi·p_ins+Ws·p_shl)`, scaled by `drive = max(p_ins,p_shl)` — so the go-high and go-low targets can no longer partly cancel. The insect "calm" gate now reads the **ambient field-median wind** (real synoptic wind), not the guaranteed-≈0 hollow, so it is no longer trivially true. | Claude |
| 014 | 2026-08-03 | **done 2026-08-09** | **Inversion-aware temperature** | **Done structurally, by data:** the field path drops the fixed 6.5 °C/km lapse entirely. Temperature is the **real interpolated field**, reconstructed per cell from a **data-driven lapse fit** (`T ~ a + b·elev` over the real lattice) — `b` is whatever the data says, negative normally or **positive under a valley inversion**, so the highest-stakes calm-autumn case gets the right sign from observations, not an assumption. | Claude |
| 015 | 2026-08-03 | **mostly done 2026-08-09** | **Remove TPI double-counting** | The field path routes terrain exposure through **one channel** (`exposure_channel`: TPI + aspect·wind blended once) that feeds both the effective wind and the refuge/shelter quality — no separate `tpi_n` re-add. Residual: TPI still appears in `exposure` and (indirectly) elevation baseline; acceptable. Legacy `downscale` path unchanged. | Claude |
| 016 | 2026-08-03 | **tested 2026-08-09 — not adopted (null)** | **Rebalance the near-constant forage offset** | Implemented as `FORAGE_RELATIVE` in `score.py` (subtract the field-mean forage) and swept under select-then-evaluate CV alongside `forage off`: **never selected on any train fold; no out-of-sample gain** on this sample → left off per the Phase-6 gate. The flag stays for retesting when more data arrives (or when IDEA 007 gives forage real within-alpine variation). | Claude |
| 017 | 2026-08-03 | **done 2026-08-09 (display)** | **Rank/percentile transform for the displayed surface** | `viz/render.py` now colours the wash by the **percentile rank** of the (smoothed) score over the field (`RANK_WASH`) — a lone outlier can no longer stretch the colour scale, contrast is stable day-to-day, and the legend reads as what it is: "how this ground ranks against the rest of the field today". Scores/CSVs untouched (min-max kept there); AUC metrics were always rank-based and are unaffected. | Claude |
| 018 | 2026-08-03 | **first sweep 2026-08-09 — all null so far** | **Validate the assumption thresholds** | First CV sweep ran via `cv_validate.py` CANDIDATES: `COLD_T_HI=5`, `WET_MM_FULL=2.5`, `ASPECT/TPI_EXPOSURE_W` 0.7/0.3 and 0.3/0.7, `EXPOSURE_SLOPE_FULL_DEG=10` — **none selected on any train fold; primary-gate selection Δ +0.014 (noise)** → all left at defaults. Honest reading: 32 confident reports can't resolve these; retest as the validation set grows. Candidates remain in the harness. | Claude |
| 019 | 2026-08-09 | **in progress (human task, 3/8 resolved)** | **Pin the vague "i området X" reports** | The reporter-error audit showed bare `at-landmark` reports score ~0.34 because the geocoder places the herd ON the named valley/lake. **2026-08-09: the pin loader now honours confirmation words** ('alright', 'ok', …) as "pin at my assumed position" — that resolved 3 of the 8 (Fellingskroken, Nørdre, Nordre) which were being silently discarded. **5 remain** (listed in `docs/validation_report.md`: Kollongen ×3-ish groups, Skarvedalen, plus 'unsure' rows) → human pins them in `manual_positions.csv`; `make_pin_template.py` is now **merge-safe** (never overwrites pins) and prints NEEDS-PIN flags. Also logged: zone-BEST (max-in-radius) evaluation tested and **rejected** (AUC 0.468 — flattens the field). | Claude |
| 020 | 2026-08-09 | **done 2026-08-09** | **Compound landmark names: direction-prefix merge in the parser** | `scrape/parse.py` now joins a direction-adjective prefix (`Nordre/Nørdre/Søre/Søndre/Øvre/Nedre/Store/Vesle/Austre/Vestre/Ytre/Indre/Midtre`) to the following capitalised word in both the direction-phrase regex and the token scan, and blocks the bare adjectives as standalone landmarks. Re-parse (same 236 rows): 8 compounds formed (Nørdre Løyftet, Nordre Vigga, Nørdre Dalen, Nørdre Svarthaugen, Nordre Skarvehøe, Søre Døkte/Døkti/Løyfte), replacing 11 fragments; **SSR resolves 5 of them `ok` directly** — positions fixed at the source. Validation impact: all-reports effort-matched AUC **0.598→0.620**, whole-field 0.508→0.542, vague-tier 0.368→0.495; primary gate 0.634→**0.641** (88% folds). | Claude |

## What lies ahead (close-out review, 2026-08-09)

Phase 6 closes with the model **above chance under the honest gate** (CV AUC 0.641,
88% of folds, n=30 position-confident) on real per-cell weather, a readable map, a
gathered `output/` tree, a full analysis folder, and an expansion plan
([`docs/expansion_plan.md`](docs/expansion_plan.md)) with the area-config seam
implemented. The next steps, in recommended order:

1. **Grow the trusted validation set** — the single biggest lever. Pin the 5
   NEEDS-PIN rows (019), review the `Nordre Skarvehøe` fuzzy match, recover the
   2022 free-prose season (002), harvest the 2026 season when it starts.
2. **Re-run the CV sweeps** (018) once n grows — the current nulls are
   sample-size-limited, not proof the thresholds are right.
3. **Refine the effort correction** with a per-report accessibility covariate (022).
4. **Port to a second area** (021) per the expansion plan — Snøhetta/Dovre is the
   lowest-effort pilot and the true test of the architecture.
5. Quality-of-life: snow layer (023), Frost station weather (011), class-weighted
   disturbance (005), within-alpine forage variation (007), web map (024).

| ID | Date | Status | Idea | Rationale | By |
|----|------|--------|------|-----------|----|
| 021 | 2026-08-09 | proposed | **Second-area pilot (Snøhetta/Dovre) + behaviour profile in the area config** | The expansion seam (`config/area.json` + `reindeer/area.py`) is implemented and wired (grid, gazetteer anchor, elevation anchors, weather lattice). Next: move the `model/score.py` behaviour constants into a `behaviour` block of the area config so profiles ship as data, then stand up Snøhetta/Dovre per `docs/expansion_plan.md` Tier A — same species, same national data sources, tests pure portability | Claude |
| 022 | 2026-08-09 | proposed | **Per-report accessibility covariate** | The effort-matched background is a global correction; the disturb-off variant still holds a small selection edge, meaning residual confound. Attach each report's own accessibility (distance from its cell to the nearest trailhead/road) as a covariate in the evaluation so the correction is per-report, not distribution-level | Claude |
| 023 | 2026-08-09 | proposed | **Snow-cover layer (senorge.no / Sentinel-2)** | The brief lists lingering snowfields as insect refuges + cooling spots; the layer was deferred through all phases. Most valuable for the early (warm) season; low priority for the autumn observation window the validation covers | Claude |
| 024 | 2026-08-09 | proposed | **Interactive web map (folium/leaflet)** | The static PNG is now readable and complete; an interactive layer (zoom, toggle overlays, tap a zone for its reason) would help in-field use on a phone. Purely presentational — no model change | Claude |
