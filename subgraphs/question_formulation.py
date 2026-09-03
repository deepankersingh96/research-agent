from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing_extensions import TypedDict

from state import QuestionFormulationState
from utils import render_yaml_prompt

# Define prompts
prompt_research_brief = "./prompts/question_formulation/research_brief.yaml"
prompt_question_formulation = "./prompts/question_formulation/question_formulation.yaml"


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


class QuestionFormulation:
    def __init__(self, model_name, max_user_clarification_count, openai_key):

        self.model = init_chat_model(
            model_name, model_provider="openai", api_key=openai_key, temperature=0
        )
        self.max_user_clarification_count = max_user_clarification_count

    def question_formulation(self, state: dict):
        """Gather information from the user"""
        result = self.model.with_structured_output(QuestionFormulationResult).invoke(
            [
                SystemMessage(
                    content=render_yaml_prompt(
                        prompt_question_formulation,
                        user_clarification_count=state.get(
                            "user_clarification_count", 0
                        ),
                        max_count=self.max_user_clarification_count,
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

    def researh_brief_generation(self, state: dict):
        """Generate initial research formuation in structured output"""
        brief = self.model.with_structured_output(ResearchBrief).invoke(
            [SystemMessage(content=render_yaml_prompt(prompt_research_brief))]
            + state["messages"]
        )

        return {"research_brief": brief, "llm_call": state.get("llm_call", 0) + 1}

    def clarification_node(self, state: dict):
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

    def should_continue(
        self,
        state: QuestionFormulationState,
        max_user_clarification_count: int,
    ) -> Literal["clarification_node", "researh_brief_generation"]:
        """Decide whether to ask for clarification or generate the brief."""
        if (
            state.get("needs_clarification", False)
            and state.get("user_clarification_count", 0) < max_user_clarification_count
        ):
            return "clarification_node"
        return "researh_brief_generation"

    def build_question_formulation_subgraph(
        self,
    ):
        # Build workflow
        agent_builder = StateGraph(QuestionFormulationState)

        # Add nodes
        agent_builder.add_node("question_formulation", self.question_formulation)
        agent_builder.add_node(
            "researh_brief_generation", self.researh_brief_generation
        )
        agent_builder.add_node("clarification_node", self.clarification_node)

        # Add edges
        agent_builder.add_edge(START, "question_formulation")
        agent_builder.add_conditional_edges(
            "question_formulation",
            self.should_continue,
            ["clarification_node", "researh_brief_generation"],
        )
        agent_builder.add_edge("clarification_node", "question_formulation")
        agent_builder.add_edge("researh_brief_generation", END)

        return agent_builder
