from __future__ import annotations

from pathlib import Path
from typing import Any

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
