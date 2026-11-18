from fastapi.testclient import TestClient

from tests.test_bm25 import DOCS
from vilegal.api import create_app
from vilegal.search import SearchEngine, best_snippet

IDS = ["a+1", "a+2", "b+1", "b+2"]
TEXTS = [f"Điều {i + 1}. Tiêu đề\n{t}" for i, t in enumerate(DOCS)]


def test_best_snippet_finds_the_densest_window():
    text = " ".join(["x"] * 50 + ["giấy", "phép", "lái", "xe"] + ["y"] * 50)
    snip = best_snippet(text, "giấy phép lái xe", width=10)
    assert "giấy phép lái xe" in snip and snip.startswith("…") and snip.endswith("…")


def test_search_api():
    engine = SearchEngine(IDS, TEXTS)
    with TestClient(create_app(engine)) as c:
        body = c.get("/search", params={"q": "đăng ký kết hôn", "k": 3}).json()
        assert body["hits"][0]["doc_id"] == "b+1"
        assert [h["rank"] for h in body["hits"]] == list(range(1, len(body["hits"]) + 1))
        assert c.get("/search", params={"q": "x"}).status_code == 422
        assert c.get("/health").json()["documents"] == 4
        assert "<html" in c.get("/").text.lower()
