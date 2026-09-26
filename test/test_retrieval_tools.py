import unittest
from unittest.mock import patch

from research_agent.state import Retrieval
from research_agent.tools.retrieval_tools import _normalize_openalex_work, open_alex_search


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


class OpenAlexNormalizationTests(unittest.TestCase):
    def test_normalizes_openalex_work_and_prefers_best_oa_location(self):
        work = {
            "id": "https://openalex.org/W123",
            "title": "A retrieval paper",
            "doi": "https://doi.org/10.1000/example",
            "abstract_inverted_index": {"retrieval": [1], "A": [0], "paper": [2]},
            "authorships": [
                {"author": {"display_name": "Ada Lovelace"}},
                {"author": {"display_name": "Grace Hopper"}},
            ],
            "topics": [{"display_name": "Information retrieval"}],
            "best_oa_location": {
                "landing_page_url": "https://arxiv.org/abs/1234.5678",
                "pdf_url": "https://arxiv.org/pdf/1234.5678",
            },
            "primary_location": {
                "landing_page_url": "https://publisher.example/paper",
                "pdf_url": "https://publisher.example/paper.pdf",
            },
        }

        retrieval = _normalize_openalex_work(work)

        self.assertEqual(retrieval["summary"], "A retrieval paper")
        self.assertEqual(retrieval["source_url"], "https://arxiv.org/abs/1234.5678")
        self.assertEqual(retrieval["pdf_url"], "https://arxiv.org/pdf/1234.5678")
        self.assertEqual(retrieval["authors"], ["Ada Lovelace", "Grace Hopper"])
        self.assertEqual(retrieval["categories"], ["Information retrieval"])
        self.assertEqual(retrieval["short_id"], "W123")
        Retrieval.model_validate(retrieval)

    def test_handles_missing_optional_openalex_fields(self):
        retrieval = _normalize_openalex_work({"id": "https://openalex.org/W456"})

        self.assertEqual(
            retrieval,
            {
                "title": "",
                "summary": "",
                "source_url": "https://openalex.org/W456",
                "authors": [],
                "categories": [],
                "pdf_url": "",
                "short_id": "W456",
            },
        )
        Retrieval.model_validate(retrieval)

    @patch("research_agent.tools.retrieval_tools.requests.get")
    def test_search_returns_normalized_retrievals(self, mock_get):
        mock_get.return_value = FakeResponse(
            {"results": [{"id": "https://openalex.org/W789", "title": "Test work"}]}
        )

        retrievals = open_alex_search.invoke({"query": "test", "max_results": 5})

        self.assertEqual(retrievals[0]["short_id"], "W789")
        self.assertEqual(mock_get.call_args.kwargs["params"]["per_page"], 5)


if __name__ == "__main__":
    unittest.main()
