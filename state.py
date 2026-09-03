# state.py
from typing import Annotated, NotRequired
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
import operator

class BaseState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_call: int

class QuestionFormulationState(BaseState, total=False):
    user_clarification_count: int
    needs_clarification: bool
    clarification_question: str
    research_brief: dict

class AgentState(QuestionFormulationState, total=False):
    expanded_queries: list[str]