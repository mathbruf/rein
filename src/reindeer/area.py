"""Study-area configuration (the expansion seam — see docs/expansion_plan.md).

Everything area-specific that the pipeline needs lives in ONE tracked JSON file,
`config/area.json` (override with the REINDEER_AREA_CONFIG environment variable).
Modules read the `AREA` singleton instead of hard-coding Lordalen values, so
porting the model to another field / villreinområde / terrain starts by writing a
new area file — not by editing source.

What the config does NOT carry (yet): the behavioural profile (regime weights &
thresholds in model/score.py are species/terrain expert judgement) and the
sightings scraper (each area has its own reporting format). Both are steps in the
expansion plan.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = _ROOT / "config" / "area.json"


@dataclass(frozen=True)
class AreaConfig:
    name: str
    region: str
    crs_epsg: int
    cell_size_m: float
    boundary_kml: Path
    primary_field: str
    buffer_field: str | None
    anchor_east: float
    anchor_north: float
    gazetteer_radius_km: float
    elev_lo_m: float
    elev_hi_m: float
    weather_lattice_forecast_m: float
    weather_lattice_archive_m: float
    dem_epsg: int

    @property
    def anchor(self) -> tuple[float, float]:
        return (self.anchor_east, self.anchor_north)


def load_area(path: Path | None = None) -> AreaConfig:
    p = Path(path or os.environ.get("REINDEER_AREA_CONFIG", DEFAULT_CONFIG))
    d = json.loads(p.read_text(encoding="utf-8"))
    kml = Path(d["boundary_kml"])
    if not kml.is_absolute():
        kml = _ROOT / kml
    return AreaConfig(
        name=d["name"], region=d.get("region", ""),
        crs_epsg=int(d["crs_epsg"]), cell_size_m=float(d["cell_size_m"]),
        boundary_kml=kml,
        primary_field=d["primary_field"], buffer_field=d.get("buffer_field"),
        anchor_east=float(d["anchor_east"]), anchor_north=float(d["anchor_north"]),
        gazetteer_radius_km=float(d["gazetteer_radius_km"]),
        elev_lo_m=float(d["elev_lo_m"]), elev_hi_m=float(d["elev_hi_m"]),
        weather_lattice_forecast_m=float(d["weather_lattice_forecast_m"]),
        weather_lattice_archive_m=float(d["weather_lattice_archive_m"]),
        dem_epsg=int(d.get("dem_epsg", d["crs_epsg"])),
    )


AREA = load_area()
