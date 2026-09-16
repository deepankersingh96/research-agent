# state.py
from typing import Annotated, NotRequired, Literal
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
import operator


GeneralResearchDomain = Literal[
    "technology_and_computer_science",  # Software, AI, cybersecurity, robotics
    "healthcare_and_medicine",          # Clinical trials, public health, nursing
    "social_sciences",                  # Psychology, education, sociology, policy
    "business_and_management",          # Economics, supply chain, HR, marketing
    "natural_and_physical_sciences",    # Physics, chemistry, environmental science
    "humanities_and_arts"               # History, literature, philosophy, design
]

class PICOCQuestion(TypedDict):
    population: list[str]
    intervention: list[str]
    comparison: list[str]
    outcome: list[str]
    context: list[str]


class FormulatedQuestion(TypedDict):
    domain: GeneralResearchDomain
    review_type: Literal['systematic', 'narrative', 'scoping', 'rapid', 'umbrella']
    framework: Literal['PICO', 'PICOC', 'PCC']
    question: PICOCQuestion


class BaseState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_call: int
    formulated_question: FormulatedQuestion


# TODO: this is an internal state and should be moved to question formulation code.
class QuestionFormulationState(BaseState, total=False):
    user_clarification_count: int
    needs_clarification: bool
    clarification_question: str
    formulated_question: FormulatedQuestion


    