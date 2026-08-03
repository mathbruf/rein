"""Phase 1: fetch & cache villreinutvalet.no/jaktinfo posts.

Etiquette (see CLAUDE.md): identify the client, go slow, cache locally so we never
re-fetch a post we already have. The site belongs to a small local committee.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE = "https://www.villreinutvalet.no"
INDEX = f"{BASE}/jaktinfo"
USER_AGENT = "reindeer-heatmap/0.1 (personal hunting research)"
DEFAULT_DELAY = 1.5  # seconds between requests — be gentle

# Repo-root-relative cache locations.
_ROOT = Path(__file__).resolve().parents[3]
RAW_DIR = _ROOT / "data" / "raw"
POSTS_DIR = RAW_DIR / "posts"


@dataclass(frozen=True)
class PostLink:
    date: str  # "DD.MM.YYYY" as shown on the index
    url: str   # absolute post URL
    slug: str  # last path segment, used as cache filename


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def crawl_index(session: requests.Session | None = None,
                delay: float = DEFAULT_DELAY,
                max_pages: int | None = None) -> list[PostLink]:
    """Walk the `?offset=` 'Eldre innlegg' chain, collecting every post link.

    Index pages are small; we always fetch them fresh to discover the full archive.
    """
    session = session or _session()
    url = INDEX
    seen: dict[str, PostLink] = {}
    pages = 0
    while url:
        resp = session.get(url, timeout=30)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        for a in soup.select("a.blog-more-link"):
            href = a.get("href", "")
            if not href:
                continue
            t = a.find_previous("time", class_="blog-date")
            slug = href.rstrip("/").rsplit("/", 1)[-1]
            seen.setdefault(slug, PostLink(
                date=t.get_text(strip=True) if t else "",
                url=BASE + href if href.startswith("/") else href,
                slug=slug,
            ))
        pages += 1
        nxt = soup.select_one("a[rel=next]")
        if not nxt or (max_pages and pages >= max_pages):
            break
        href = nxt["href"]
        url = BASE + href if href.startswith("/") else href
        time.sleep(delay)
    return list(seen.values())


def fetch_post(link: PostLink, session: requests.Session | None = None,
               delay: float = DEFAULT_DELAY) -> Path:
    """Fetch one post into data/raw/posts/<slug>.html. Skips if already cached."""
    session = session or _session()
    POSTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = POSTS_DIR / f"{link.slug}.html"
    if dest.exists() and dest.stat().st_size > 0:
        return dest  # already have it — resumable
    resp = session.get(link.url, timeout=30)
    resp.raise_for_status()
    dest.write_text(resp.text, encoding="utf-8")
    time.sleep(delay)
    return dest
