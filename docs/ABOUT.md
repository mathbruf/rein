# About this project — how it should be viewed

## What this is

This is a **research and learning analysis of wild reindeer (*villrein*) movement** in the
Lordalen study area of Reinheimen villreinområde, Norway. It is purely analytical: the goal
is to understand **what makes reindeer move** — and, through that, what is most important
in an animal's daily life.

The central question is simple to state and hard to answer:

> Of all the pressures acting on a wild reindeer on a given day — weather, insects, food,
> terrain, human presence — **which ones actually decide where the animals go?**

The project answers it by *building the hypothesis as a model and then testing it*. A
rule-based scoring function encodes an explicit, documented theory of reindeer behaviour
(shelter-seeking in cold/wet/wind, insect avoidance on warm calm days, forage value,
disturbance avoidance, a baseline preference for high ground). Every day it turns the
weather forecast plus the fixed landscape into a 0–1 probability surface over a 250 m
grid. Independently collected observation reports — never used to build or tune the model —
are then used to check whether the surface ranks real reindeer positions above chance.
Where it does, the encoded behavioural logic has explanatory power; where it does not,
the theory is wrong or incomplete, and that is recorded honestly.

## What the analysis has found so far

The validation work is itself the learning output. Highlights:

- **Weather and shelter dominate the daily decision** in the autumn observation window:
  the shelter driver (cold, wind, rain) fires on nearly every validated day, while the
  insect driver is almost always off by that season. Day-to-day movement is governed more
  by weather than by insects — exactly what the local field expert described.
- **Wind direction matters physically:** modelling genuine leeward/windward shelter from
  real per-cell wind direction and slope aspect measurably improved out-of-sample ranking.
- **Human disturbance shapes the distribution** — but measuring it fairly is subtle,
  because the observation reports themselves come from accessible terrain (observer effort
  bias). Separating "animals avoid people" from "people report where people go" is one of
  the project's ongoing methodological lessons.
- **Honest nulls are kept:** threshold and forage refinements that failed to improve
  cross-validated performance were reverted and logged, not kept.

## What this is not

- **Not a tracking or locating tool.** It does not — and cannot — predict the herd's
  actual position at a given time. Reindeer move as a social herd and tomorrow depends
  heavily on where they are today, which is unknown to the model. The output is a
  probability surface expressing which ground the day's conditions *favour*, nothing more.
- **Not a management or decision system.** No output of this project should be used to
  approach, follow, or disturb the animals, nor as an operational basis for decisions
  about them. Wild reindeer are highly sensitive to human presence; the right way to use
  this work is at a desk, comparing the model's expectations against reported observations.
- **Not a finished scientific result.** The validation sample is small (a few dozen
  landmark-level reports per season, ≈1–5 km positional precision, observer-effort bias).
  The claims are kept correspondingly modest: a cross-validated, better-than-chance
  prototype — evidence that the encoded drivers are real, not a definitive model.

## Why this framing matters

Wild reindeer are one of Europe's last intact wild mountain ungulate populations, and
Norway carries a special responsibility for them. A model like this is valuable precisely
*because* it is analytical: it turns scattered, coarse observations and public weather and
terrain data into a testable statement about what the animals need — wind-cooled ridges on
warm days, sheltered slopes in storms, undisturbed ground always. Understanding those
needs, and being honest about how well we can measure them, is the whole point.

## Data etiquette

The observation reports come from `villreinutvalet.no`, a site run by a small local
committee. The scraper is deliberately gentle (low rate, local caching, an identifying
User-Agent), and a polite request to the committee for structured data is documented as
the preferred path. All weather, elevation and land-cover data come from open public
sources (MET Norway / Open-Meteo, Kartverket, NIBIO) under their respective terms.
