"""Quick manual test for all three trained models.

Usage
-----
    python test_models.py                        # test with 3 default users, top-5
    python test_models.py --users 1 42 100       # specific user IDs
    python test_models.py --k 10                 # top-10 recommendations
    python test_models.py --model svd            # one model only
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--users", nargs="+", type=int, default=None, help="user IDs to test (default: first 3 from train)")
    ap.add_argument("--k", type=int, default=5, help="number of recommendations (default: 5)")
    ap.add_argument("--model", choices=["popularity", "svd", "hybrid"], default=None, help="test one model only")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    train = pd.read_parquet(root / "data/processed/train.parquet")
    items = pd.read_parquet(root / "data/processed/items_clean.parquet")
    title = items.set_index("item_id")["title"]

    user_ids = args.users or train["user_id"].unique()[:3].tolist()
    model_names = [args.model] if args.model else ["popularity", "svd", "hybrid"]

    print(f"User IDs : {user_ids}")
    print(f"Top-K    : {args.k}")
    print()

    for name in model_names:
        model = joblib.load(root / f"models/{name}.joblib")
        recs = model.recommend(user_ids, k=args.k)
        ratings = model.predict_rating(user_ids, user_ids)
        print(f"{'=' * 50}")
        print(f"Model: {name}")
        print(f"{'=' * 50}")
        for uid, rec, r in zip(user_ids, recs, ratings):
            titles = [title.get(i, f"item_{i}") for i in rec]
            print(f"  user {uid} (predicted rating: {r:.2f})")
            for rank, t in enumerate(titles, 1):
                print(f"    {rank}. {t}")
        print()


if __name__ == "__main__":
    main()