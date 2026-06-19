# Ideas Backlog

Statuses: proposed | accepted | done | rejected
Agents add ideas here as `proposed`; the human promotes them to ROADMAP.

| ID | Date | Status | Idea | Rationale | By |
|----|------|--------|------|-----------|----|
| 001 | YYYY-MM-DD | example | Include solar radiation / aspect as a heat-load proxy | May sharpen the warm-day "climb to cool exposed ground" signal | example |
| 002 | 2026-06-12 | proposed | Free-prose extractor for pre-2023 (2022) jaktinfo posts | 2022 posts are narrative (no `Region:` lines); a sentence-level NLP pass could recover ~1 more season of validation sightings, at the cost of noisier landmark/count extraction | Claude |
| 003 | 2026-06-19 | proposed | Upgrade DTM 50 m → 10 m for terrain derivatives | TPI/ruggedness/slope at 250 m cells would be sharper from a 10 m DTM (25 vs 625 source pixels/cell); needs tiled WCS requests (one request would be ~29 M px). 50 m is adequate for v0 | Claude |
| 004 | 2026-06-19 | proposed | Elevation-aware temperature in the scorer | Apply a lapse rate (~-0.6 °C/100 m) to the area forecast so high cells are modeled colder than the valley reading; sharpens the per-cell insect/thermal vs shelter regime split instead of using one area-wide temp | Claude |
