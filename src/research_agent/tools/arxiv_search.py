from langchain.tools import tool
import arxiv


@tool
def search_arxiv(query: str, max_results:int = 10):
    """
    Given a arxiv supported search string, 
    perform search in the arxiv database and return the title, 
    abstract and metadata of the retrieved papers.
    """
    client =  arxiv.Client()

    search = arxiv.Search(
        query=query, 
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    return search

