from langchain.tools import tool
import arxiv


@tool
def search_arxiv(query: str, max_results: int = 10):
    """
    Given a arxiv supported search string,
    perform search in the arxiv database and return the title,
    abstract and metadata of the retrieved papers.
    """
    client = arxiv.Client()

    search = arxiv.Search(
        query=query, max_results=max_results, sort_by=arxiv.SortCriterion.Relevance
    )

    results = client.results(search)

    retrievals = [
        {
            "title": res.title,
            "summary": res.summary,
            "source_url": res.source_url,
            "authors": res.authors,
            "categories": res.categories,
            "pdf_url": res.pdf_url,
            "short_id": res.get_short_id(),
        }
        for res in results
    ]

    return retrievals


if __name__ == "__main__":
    ret = search_arxiv.invoke(
        {
            "query":'rag system for test generation',
            "max_results":10,
        }
       
    )
    print (ret)
