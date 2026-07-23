import unittest

from research.contracts import ExternalSource, deduplicate_sources, normalize_url


class ResearchContractTests(unittest.TestCase):
    def test_normalize_url_removes_tracking_and_fragment(self):
        normalized = normalize_url("HTTPS://Example.COM/path/?utm_source=x&b=2&a=1#part")
        self.assertEqual(normalized, "https://example.com/path?a=1&b=2")

    def test_deduplicate_sources_assigns_stable_ids(self):
        sources = [
            ExternalSource(
                title="A",
                url="https://example.com/item?utm_source=x",
                provider="tavily",
                query="q",
            ),
            ExternalSource(
                title="B",
                url="https://EXAMPLE.com/item",
                provider="tavily",
                query="q",
            ),
            ExternalSource(
                title="C",
                url="https://example.org/other",
                provider="tavily",
                query="q",
            ),
        ]
        result = deduplicate_sources(sources)
        self.assertEqual([item.source_id for item in result], ["source_1", "source_2"])
        self.assertEqual([item.title for item in result], ["A", "C"])


if __name__ == "__main__":
    unittest.main()
