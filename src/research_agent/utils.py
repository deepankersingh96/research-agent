from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from docling.datamodel.base_models import DocumentStream
from docling.document_converter import DocumentConverter
from jinja2 import Environment, StrictUndefined


_JINJA_ENV = Environment(
    autoescape=False,
    keep_trailing_newline=True,
    undefined=StrictUndefined,
)


def render_yaml_prompt(path: str | Path, **kwargs: Any) -> str:
    """
    Read a YAML prompt template and render it with Jinja2.
    """
    template = Path(path).read_text(encoding="utf-8")
    return _JINJA_ENV.from_string(template).render(**kwargs)


def convert_pdf_url_to_markdown(
    url: str, openalex_api_key: str | None = None
) -> str:
    """Download a PDF URL, convert it with Docling, and return Markdown."""
    if url == "":
        return "ERR:No pdf_url for the given document."

    headers = {}
    if openalex_api_key:
        headers["Authorization"] = f"Bearer {openalex_api_key}"

    response = requests.get(url, headers=headers, timeout=30)
    response.raise_for_status()
    doc_stream = DocumentStream(
        name="document.pdf", stream=BytesIO(response.content)
    )
    result = DocumentConverter().convert(doc_stream)
    return result.document.export_to_markdown()
