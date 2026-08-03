"""Phase 1 harvest: crawl jaktinfo, cache posts, parse to a tidy observations table.

Usage (from repo root, venv active):
    python scripts/harvest_jaktinfo.py --max-pages 1     # newest 20 posts only
    python scripts/harvest_jaktinfo.py                   # full archive

Outputs:
    data/raw/posts/<slug>.html            cached raw posts (resumable)
    data/interim/observations.csv         one row per spatial claim
    data/interim/parsing_report.txt       posts/lines that couldn't be parsed
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make src/ importable when run as a script.
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

import pandas as pd  # noqa: E402

from reindeer.scrape.fetch import crawl_index, fetch_post, _session  # noqa: E402
from reindeer.scrape.parse import parse_post, to_rows  # noqa: E402

INTERIM = _ROOT / "data" / "interim"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-pages", type=int, default=None,
                    help="limit index pages crawled (20 posts/page). Default: all.")
    args = ap.parse_args()

    INTERIM.mkdir(parents=True, exist_ok=True)
    session = _session()

    print("Crawling index (Eldre innlegg chain)...", flush=True)
    links = crawl_index(session, max_pages=args.max_pages)
    print(f"  found {len(links)} posts", flush=True)

    all_rows: list[dict] = []
    report: list[str] = []
    for i, link in enumerate(links, 1):
        path = fetch_post(link, session)
        parse = parse_post(path.read_text(encoding="utf-8"), link.url)
        rows = to_rows(parse)
        all_rows.extend(rows)
        if parse.unparsed_lines:
            report.append(f"{link.date} {link.url}")
            report.extend(f"    ? {l}" for l in parse.unparsed_lines)
        print(f"  [{i}/{len(links)}] {link.date}  {len(rows)} obs", flush=True)

    df = pd.DataFrame(all_rows)
    # Serialise list/dict columns so the CSV stays readable.
    for col in ("landmark_phrases", "direction_hints", "count_estimate"):
        if col in df:
            df[col] = df[col].apply(lambda v: json.dumps(v, ensure_ascii=False))
    csv_path = INTERIM / "observations.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    report_path = INTERIM / "parsing_report.txt"
    report_path.write_text(
        ("Posts/lines that could not be parsed:\n\n" + "\n".join(report))
        if report else "All posts parsed; no unparsed observation lines.\n",
        encoding="utf-8",
    )

    print(f"\nWrote {len(df)} observation rows -> {csv_path}")
    print(f"Parsing report -> {report_path}"
          f"  ({len(report)} flagged)" if report else f"Parsing report -> {report_path}")
    if not df.empty:
        print(f"Date range: {df['date'].min()} .. {df['date'].max()}")
        print("Regions:", ", ".join(f"{r}({n})" for r, n
                                     in df['region'].value_counts().items()))


if __name__ == "__main__":
    main()
