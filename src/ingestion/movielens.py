"""MovieLens-100K download + parsing into the pipeline's common raw schema."""
from __future__ import annotations

import shutil
import ssl
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

GENRES = ["unknown", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
          "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
          "Romance", "Sci-Fi", "Thriller", "War", "Western"]


def _download(url: str, dest: Path) -> None:
    try:  # certifi avoids the classic "CERTIFICATE_VERIFY_FAILED" on python.org builds for macOS
        import certifi
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:
        ctx = ssl.create_default_context()
    with urllib.request.urlopen(url, context=ctx, timeout=60) as resp, open(dest, "wb") as out:
        shutil.copyfileobj(resp, out)


def ensure_movielens(raw_dir: Path, url: str) -> Path:
    """Return the extracted ``ml-100k`` folder, downloading it only if necessary.

    You can also drop ``ml-100k.zip`` (or the extracted ``ml-100k`` folder) into
    ``data/raw`` yourself - the download is then skipped.
    """
    folder = raw_dir / "ml-100k"
    if (folder / "u.data").exists():
        return folder
    zip_path = raw_dir / "ml-100k.zip"
    if not zip_path.exists():
        _download(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    if not (folder / "u.data").exists():
        raise FileNotFoundError(f"{folder}/u.data not found after extracting {zip_path}")
    return folder


def parse_movielens_100k(folder: Path):
    """Parse the three MovieLens files into (users, items, interactions) DataFrames."""
    folder = Path(folder)
    interactions = pd.read_csv(folder / "u.data", sep="\t",
                               names=["user_id", "item_id", "rating", "timestamp"])
    users = pd.read_csv(folder / "u.user", sep="|",
                        names=["user_id", "age", "gender", "occupation", "zip_code"],
                        dtype={"zip_code": str})
    raw_items = pd.read_csv(folder / "u.item", sep="|", header=None, encoding="latin-1",
                            names=["item_id", "title", "release_date", "video_release_date",
                                   "imdb_url"] + GENRES)
    flags = raw_items[GENRES].to_numpy()
    genre_strings = ["|".join(g for g, flag in zip(GENRES, row) if flag) or "unknown" for row in flags]
    items = pd.DataFrame({
        "item_id": raw_items["item_id"],
        "title": raw_items["title"],
        "release_date": raw_items["release_date"],
        "genres": genre_strings,
    })
    return users, items, interactions
