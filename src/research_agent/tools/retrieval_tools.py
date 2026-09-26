from langchain.tools import tool
import arxiv
from dotenv import load_dotenv
import requests
import os
from typing import Any, Optional

load_dotenv()


def _reconstruct_abstract(
    abstract_inverted_index: Optional[dict[str, list[int]]],
) -> str:
    """Convert OpenAlex's positional abstract representation to plain text."""
    if not abstract_inverted_index:
        return ""

    words_by_position: dict[int, str] = {}
    for word, positions in abstract_inverted_index.items():
        for position in positions:
            words_by_position[position] = word

    return " ".join(
        word for _, word in sorted(words_by_position.items(), key=lambda item: item[0])
    )


def _location_url(work: dict[str, Any], key: str) -> str:
    """Prefer the best OA copy, then the primary publication location."""
    for location_name in ("best_oa_location", "primary_location"):
        location = work.get(location_name) or {}
        if value := location.get(key):
            return value
    return ""


def _normalize_openalex_work(work: dict[str, Any]) -> dict[str, Any]:
    """Map one OpenAlex work record to the project's provider-neutral shape."""
    source_url = (
        _location_url(work, "landing_page_url")
        or work.get("doi")
        or work.get("id", "")
    )

    return {
        "title": work.get("title") or "",
        "summary": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "source_url": source_url,
        "authors": [
            authorship["author"]["display_name"]
            for authorship in work.get("authorships") or []
            if authorship.get("author", {}).get("display_name")
        ],
        "categories": [
            topic["display_name"]
            for topic in work.get("topics") or []
            if topic.get("display_name")
        ],
        "pdf_url": _location_url(work, "pdf_url"),
        "short_id": (work.get("id") or "").rstrip("/").rsplit("/", 1)[-1],
    }


@tool
def search_arxiv(query: str, max_results: int = 10):
    """
    Given an arxiv supported search string,
    perform search in the arxiv database and return the title,
    abstract and metadata of the retrieved papers.
    """
    client = arxiv.Client(page_size=max_results, delay_seconds=4)

    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    results = client.results(search)

    retrievals = [
        {
            "title": res.title,
            "summary": res.summary,
            "source_url": res.source_url,
            "authors": [author.name for author in res.authors],  # Extract names clean
            "categories": res.categories,
            "pdf_url": res.pdf_url,
            "short_id": res.entry_id.split("/abs/")[
                -1
            ],  # Safer fallback than get_short_id()
        }
        for res in results
    ]

    return retrievals


@tool
def semantic_scholar_search(
    query: str,
    max_results: int = 100,
    fields: str = "paperId,title,abstract,url,publicationTypes,publicationDate,openAccessPdf",
    year: str = "2023-",
):
    """
    Search papers in the semantic scholar database.
    Input:
     - query : semantic scholar formatted query string
     - fields : fields to retrieve; a comma separated string of fields with NO spaces
     - year : year of publication in endash format
    """

    # Define the API endpoint URL
    url = "http://api.semanticscholar.org/graph/v1/paper/search"

    # Define the query parameters
    query_params = {
        "query": query,
        "fields": fields,
        "year": year,
        "offset": 0,
        "limit": max_results,
    }

    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    headers = {"x-api-key": api_key}

    response = requests.get(url, params=query_params, headers=headers)
    print(response.status_code, response.headers)
    print(response.text)
    response.raise_for_status()
    payload = response.json()

    return payload


@tool
def open_alex_search(
    query: str,
    max_results: int = 100,
    fields: str = "paperId,title,abstract,url,publicationTypes,publicationDate,openAccessPdf",
    year: str = "2023-",
):
    """
    Search papers in the Open Alex database.
    Input:
     - query : plain-text OpenAlex search query
     - max_results : maximum number of relevance-ranked works to return

    ``fields`` and ``year`` are retained for compatibility with the existing
    tool interface. OpenAlex returns its standard work representation, which
    this tool normalizes below.
    """

    # Define the API endpoint URL
    url = "https://api.openalex.org/works"

    # Define the query parameters
    query_params = {
        "search": query,
        "per_page": max_results,
    }
    if api_key := os.environ.get("OPENALEX_API_KEY"):
        query_params["api_key"] = api_key

    response = requests.get(url, params=query_params, timeout=30)
    response.raise_for_status()
    results = response.json().get("results", [])

    return [_normalize_openalex_work(work) for work in results]


if __name__ == "__main__":
    from pprint import pprint

    clean_query = "retrieval augmented generation"

    retrievals = open_alex_search.invoke({"query": clean_query, "max_results": 5})
    pprint(retrievals)
