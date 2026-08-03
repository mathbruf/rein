"""Phase 1: parse a cached jaktinfo post into structured observation rows.

A post is: intro prose -> "Observasjonar etter dagens jakt:" -> one line per region
-> closing prose. We emit ONE ROW PER SPATIAL CLAIM (sentence) within each region line,
keeping all regions (tagged). See ROADMAP.md Phase 1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict

from bs4 import BeautifulSoup

# A region-prefixed observation line. The region is one of the known område names
# (optionally with a sub-area qualifier), followed by ':' OR ' - '/' – ' then text.
# We scan EVERY body line for this — older seasons (2023) omit the "Observasjonar"
# header but still use region-prefixed lines, so we must not depend on a header.
_SUBAREA = r"(?:\s+(?:søraust|sørvest|sør|nordaust|nordvest|nord|austre|vestre|nordre|søndre|aust|vest|almenning))*"
_REGION_RE = re.compile(
    r"^((?:Reinheimen|Breheimen|Skjåk|Lesja|Lordalen)" + _SUBAREA + r")"
    r"\s*[:\-–]\s+(.+)$",
    re.I,
)
# Count phrase: optional qualifier, optional number, an animal unit (incl. common
# compounds like "storbukkar", "bukkeflokk", "fostringsflokk").
_COUNT_RE = re.compile(
    r"(minimum|min\.?|ca\.?|om lag|rundt|opp mot|mange|mykje|fleire|nokre|nokon|liten)?\s*"
    r"(\d+)?\s*(\w*(?:dyr|flokk|bukk|simle|kalv|kviger|kyr)\w*)\b",
    re.I,
)
# Direction phrase: optional bearing + a spatial relation + the landmark token.
_DIR_RE = re.compile(
    r"((?:nord|sør|aust|vest|nordaust|nordvest|søraust|sørvest)(?:over)?)?\s*"
    r"\b(for|frå|fra|mot|innanfor|innan|i området|sør for|nord for|aust for|vest for|ved|over)\s+"
    r"([A-ZÆØÅ][\wæøåÆØÅ]+(?:/[A-ZÆØÅ][\wæøåÆØÅ]+)*)",
    re.I,
)
# Capitalised tokens that are NOT place names (sentence openers / quantity words).
_NOT_PLACE = {
    "Minimum", "Fleire", "Mykje", "Nokre", "Nokon", "Liten", "Litt", "Ein", "Eit",
    "Det", "Dei", "Vi", "Her", "Der", "Ingen", "Om", "Ca", "Min", "Hovudmassa",
    "Storparten", "Resten", "Ellers", "Elles", "Same", "Framleis", "Truleg",
    "Stor", "Mange", "Mindre", "Store", "Små", "Ein", "Også",
    # not place names: "scattered", "from", directions used as sentence openers
    "Spredt", "Spredte", "Spreidd", "Spreidde", "Frå", "Fra", "Nord", "Sør",
    "Aust", "Vest", "Nordaust", "Nordvest", "Søraust", "Sørvest",
    # område / region names — not pinpoint landmarks
    "Reinheimen", "Breheimen", "Reinheimen-Breheimen",
}


@dataclass
class Observation:
    date: str
    region: str
    observation_text: str
    count_estimate: dict | None
    landmark_phrases: list[str] = field(default_factory=list)
    direction_hints: list[dict] = field(default_factory=list)


@dataclass
class PostParse:
    source_url: str
    post_title: str
    date: str
    observations: list[Observation] = field(default_factory=list)
    unparsed_lines: list[str] = field(default_factory=list)


def _count(sentence: str) -> dict | None:
    m = _COUNT_RE.search(sentence)
    if not m:
        return None
    return {
        "raw": m.group(0).strip(),
        "qualifier": (m.group(1) or "").strip().lower() or None,
        "number": int(m.group(2)) if m.group(2) else None,
        "unit": m.group(3).lower(),
    }


def _landmarks(sentence: str, dir_hints: list[dict]) -> list[str]:
    """Real place names: those captured by direction phrases, plus other capitalised
    tokens that aren't sentence-initial and aren't known non-places."""
    out: list[str] = []
    for d in dir_hints:
        for part in d["landmark"].split("/"):
            if part not in out:
                out.append(part)
    tokens = re.findall(r"[A-ZÆØÅ][\wæøåÆØÅ]+(?:/[A-ZÆØÅ][\wæøåÆØÅ]+)*", sentence)
    for i, tok in enumerate(tokens):
        for part in tok.split("/"):
            if i == 0 and sentence.startswith(part):
                continue  # sentence opener, e.g. "Minimum", "Fleire"
            if part in _NOT_PLACE or part in out:
                continue
            out.append(part)
    return out


def _directions(sentence: str) -> list[dict]:
    # _DIR_RE is case-insensitive for the bearing/relation, so the landmark group
    # can match lowercase words ("for tiden" -> "tiden"). Require a genuinely
    # capitalised, non-stopword landmark.
    out = []
    for (b, rel, lm) in _DIR_RE.findall(sentence):
        head = lm.split("/")[0]
        if not head[:1].isupper() or head in _NOT_PLACE:
            continue
        out.append({"bearing": (b or None), "relation": rel.lower(), "landmark": lm})
    return out


def _norm_region(label: str) -> str:
    """Collapse whitespace and Title-case the leading område word, keeping sub-area."""
    parts = label.split()
    if parts:
        parts[0] = parts[0].capitalize()
    return " ".join(parts)


def _looks_like_prose(line: str) -> bool:
    """Long, multi-sentence lines are narrative, not a single missed observation.
    We only flag short region-less lines (likely real observations) for the report."""
    return len(line) > 140 or line.count(".") >= 3


def parse_post(html: str, source_url: str) -> PostParse:
    soup = BeautifulSoup(html, "lxml")
    title_el = soup.select_one("h1.entry-title")
    title = title_el.get_text(strip=True) if title_el else ""
    meta = soup.find("meta", {"itemprop": "datePublished"})
    date = meta["content"][:10] if meta and meta.get("content") else ""

    body = soup.select_one(".blog-item-content-wrapper")
    lines = [p.get_text(" ", strip=True)
             for p in body.select("p,h1,h2,h3,li")] if body else []
    lines = [l for l in lines if l]

    result = PostParse(source_url=source_url, post_title=title, date=date)

    # Scan every body line: region-prefixed lines are the observations. We do NOT
    # require the "Observasjonar" header — 2023 posts omit it but still use
    # region-prefixed lines. Non-region lines that still mention animals are flagged
    # for the parsing report (these are the free-prose posts, mostly pre-2023).
    for line in lines:
        m = _REGION_RE.match(line)
        if not m:
            if any(u in line.lower() for u in ("dyr", "flokk", "bukk")) \
                    and not _looks_like_prose(line):
                result.unparsed_lines.append(line)
            continue
        region, text = _norm_region(m.group(1)), m.group(2).strip()
        for sent in re.split(r"(?<=[.!?])\s+", text):
            sent = sent.strip()
            if not sent:
                continue
            dirs = _directions(sent)
            count = _count(sent)
            landmarks = _landmarks(sent, dirs)
            # An observation needs animal evidence: a count, or a direction at a real
            # landmark. Pure tactical advice / weather notes have neither -> skip.
            if count is None and not dirs:
                continue
            result.observations.append(Observation(
                date=date,
                region=region,
                observation_text=sent,
                count_estimate=count,
                landmark_phrases=landmarks,
                direction_hints=dirs,
            ))
    return result


def to_rows(parse: PostParse) -> list[dict]:
    """Flatten a PostParse into CSV-friendly dict rows."""
    rows = []
    for obs in parse.observations:
        d = asdict(obs)
        d["source_url"] = parse.source_url
        d["post_title"] = parse.post_title
        rows.append(d)
    return rows
