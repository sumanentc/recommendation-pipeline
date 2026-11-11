from src.ingestion.movielens import GENRES, parse_movielens_100k


def test_parse_movielens_100k(tmp_path):
    (tmp_path / "u.data").write_text("1\t1\t5\t874965758\n1\t2\t3\t876893171\n2\t1\t4\t878542960\n")
    (tmp_path / "u.user").write_text("1|24|M|technician|85711\n2|53|F|other|94043\n")
    flags1 = ["0"] * 19; flags1[GENRES.index("Animation")] = "1"; flags1[GENRES.index("Comedy")] = "1"
    flags2 = ["0"] * 19                                              # no genre flag at all
    rows = ["1|Toy Story (1995)|01-Jan-1995||http://x|" + "|".join(flags1),
            "2|Caf\xe9 Society (1996)|01-Jan-1996||http://y|" + "|".join(flags2)]
    (tmp_path / "u.item").write_bytes(("\n".join(rows) + "\n").encode("latin-1"))

    users, items, inter = parse_movielens_100k(tmp_path)
    assert list(inter.columns) == ["user_id", "item_id", "rating", "timestamp"] and len(inter) == 3
    assert list(users.columns) == ["user_id", "age", "gender", "occupation", "zip_code"]
    assert users.loc[0, "zip_code"] == "85711"
    assert items.loc[0, "genres"] == "Animation|Comedy"
    assert items.loc[1, "genres"] == "unknown" and items.loc[1, "title"].startswith("Caf\xe9")
