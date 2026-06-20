# Ideas Backlog

Statuses: proposed | accepted | done | rejected
Agents add ideas here as `proposed`; the human promotes them to ROADMAP.

| ID | Date | Status | Idea | Rationale | By |
|----|------|--------|------|-----------|----|
| 001 | YYYY-MM-DD | example | Include solar radiation / aspect as a heat-load proxy | May sharpen the warm-day "climb to cool exposed ground" signal | example |
| 002 | 2026-06-12 | proposed | Free-prose extractor for pre-2023 (2022) jaktinfo posts | 2022 posts are narrative (no `Region:` lines); a sentence-level NLP pass could recover ~1 more season of validation sightings, at the cost of noisier landmark/count extraction | Claude |
| 003 | 2026-06-19 | proposed | Upgrade DTM 50 m → 10 m for terrain derivatives | TPI/ruggedness/slope at 250 m cells would be sharper from a 10 m DTM (25 vs 625 source pixels/cell); needs tiled WCS requests (one request would be ~29 M px). 50 m is adequate for v0 | Claude |
| 004 | 2026-06-19 | proposed | Elevation-aware temperature in the scorer | Apply a lapse rate (~-0.6 °C/100 m) to the area forecast so high cells are modeled colder than the valley reading; sharpens the per-cell insect/thermal vs shelter regime split instead of using one area-wide temp | Claude |
| 005 | 2026-06-19 | proposed | Weight disturbance by feature type | v1 disturbance treats a trunk road and a faint footpath equally (distance to nearest of any). Weight by class (trunk/secondary » track » path; manned cabin » trailhead) so busy access penalises more than a faint trail | Claude |
| 006 | 2026-06-19 | proposed | Cluster the daily top-zones into named areas | daily_map.py lists top individual 250 m cells, which are often adjacent. Cluster high-score cells into a handful of distinct zones (centroid + extent + nearest gazetteer landmark) for a cleaner "go here" list | Claude |
| 007 | 2026-06-19 | proposed | Refine forage with AR50 vegetasjonsdekke | Lordalen is 89% one arealtype (open alpine), so forage barely discriminates within the zone. The AR50 vegetasjonsdekke field (vegetated vs bare rock/scree) could separate prime lichen/grass from barren ground | Claude |
| 008 | 2026-06-20 | proposed | **Seasonal scorer profiles (autumn vs summer)** | Phase-5 validation (hunt-season sightings) shows the summer high-ground baseline + insect 'go-high' driver are anti-correlated with autumn reports; ablating both lifts AUC 0.32→0.60. Add a season switch that turns the insect driver + high-ground baseline down/off for late Aug–Sept and prefers low/mid elevation. Re-test on held-out/CV data, not the same set | Claude |
| 009 | 2026-06-20 | proposed | **Effort/accessibility covariate for validation** | The disturbance penalty hurts validation because presence-only reports are effort-biased toward accessible terrain. Add an effort covariate or restrict the presence-background comparison to huntable/accessed terrain so the disturbance rule can be judged fairly | Claude |
| 010 | 2026-06-20 | proposed | k-fold cross-validation harness | So weight changes motivated by validation are tested out-of-sample (the Phase-5 ablations used the test set and are only hypotheses). Needed before any validation-driven retuning can be trusted | Claude |
| 011 | 2026-06-20 | proposed | MET Frost as historical-weather upgrade | Phase 5 used Open-Meteo ERA5 archive (no key) instead of MET Frost. With a free Frost client ID, compare nearest-station vs ERA5 and optionally switch, or keep ERA5 for gridded coverage | Claude |
