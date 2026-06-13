from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv

import operator
import os

from utils import render_yaml_prompt

# Load dotenv
load_dotenv()
model_name = os.environ.get("OPENAI_MODEL_NAME")
openai_key = os.environ.get("OPENAI_API_KEY")

# TODO: move to a config file
max_user_clarification_count = 3

# Define prompts
prompt_research_brief = "./prompts/question_formulation/research_brief.yaml"
prompt_question_formulation = "./prompts/question_formulation/question_formulation.yaml"

# Define model and tools
model = init_chat_model(
    model_name, model_provider="openai", api_key=openai_key, temperature=0
)


# Define state
class ResearchState(TypedDict):
    user_query: str
    messages: Annotated[list[AnyMessage], operator.add]
    research_question: str
    research_brief: dict
    user_clarification_count: int
    llm_call: int
    needs_clarification: bool
    clarification_question: str


class ResearchBrief(TypedDict):
    research_question: str
    research_type: Literal[
        "fact_finding",
        "comparison",
        "trend_analysis",
        "evaluation",
        "exploration",
        "decision_support",
    ]
    scope: list[str]
    constraints: list[str]
    success_criteria: list[str]
    assumptions: list[str]


class QuestionFormulationDecision(BaseModel):
    needs_clarification: bool
    clarification_question: str | None = None


class QuestionFormulationResult(BaseModel):
    decision: QuestionFormulationDecision


# Define model node
def question_formulation(state: dict):
    """Gather information from the user"""
    result = model.with_structured_output(QuestionFormulationResult).invoke(
        [
            SystemMessage(
                content=render_yaml_prompt(
                    prompt_question_formulation,
                    user_clarification_count=state.get("user_clarification_count", 0),
                    max_count=max_user_clarification_count,
                )
            )
        ]
        + state["messages"]
    )
    return {
        "needs_clarification": result.decision.needs_clarification,
        "clarification_question": result.decision.clarification_question or "",
        "llm_call": state.get("llm_call", 0) + 1,
    }


def researh_brief_generation(state: dict):
    """Generate initial research formuation in structured output"""
    brief = model.with_structured_output(ResearchBrief).invoke(
        [SystemMessage(content=render_yaml_prompt(prompt_research_brief))]
        + state["messages"]
    )

    return {"research_brief": brief, "llm_call": state.get("llm_call", 0) + 1}


# Define clarification node
def clarification_node(state: dict):
    """Ask the user for the single most important missing detail."""
    question = state["clarification_question"]
    observation = input(f"Agent: {question}\nYour answer: ")
    return {
        "messages": [
            AIMessage(content=question),
            HumanMessage(content=observation),
        ],
        "user_clarification_count": state.get("user_clarification_count", 0) + 1,
    }


# Define end logic


def should_continue(
    state: ResearchState,
) -> Literal["clarification_node", "researh_brief_generation"]:
    """Decide whether to ask for clarification or generate the brief."""
    if (
        state.get("needs_clarification", False)
        and state.get("user_clarification_count", 0) < max_user_clarification_count
    ):
        return "clarification_node"
    return "researh_brief_generation"


# Build workflow
agent_builder = StateGraph(ResearchState)

# Add nodes
agent_builder.add_node("question_formulation", question_formulation)
agent_builder.add_node("researh_brief_generation", researh_brief_generation)
agent_builder.add_node("clarification_node", clarification_node)

# Add edges
agent_builder.add_edge(START, "question_formulation")
agent_builder.add_conditional_edges(
    "question_formulation",
    should_continue,
    ["clarification_node", "researh_brief_generation"],
)
agent_builder.add_edge("clarification_node", "question_formulation")
agent_builder.add_edge("researh_brief_generation", END)

# Compile the agent
agent = agent_builder.compile()


def main():
    query = input(f"Enter your research query: ")
    messages = [HumanMessage(content=query)]
    result = agent.invoke({"messages": messages})
    for m in result["messages"]:
        m.pretty_print()
    print(result["research_brief"])


if __name__ == "__main__":
    main()
