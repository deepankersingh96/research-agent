from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from typing_extensions import TypedDict

from research_agent.state import QuestionFormulationState, FormulatedQuestion
from research_agent.utils import render_yaml_prompt

# Define prompts
prompt_research_brief = "src/research_agent/prompts/research_brief.yaml"
prompt_question_formulation = "src/research_agent/prompts/question_formulation.yaml"


# class ResearchBrief(TypedDict):
#     research_question: str
#     research_type: Literal[
#         "fact_finding",
#         "comparison",
#         "trend_analysis",
#         "evaluation",
#         "exploration",
#         "decision_support",
#     ]
#     scope: list[str]
#     constraints: list[str]
#     success_criteria: list[str]
#     assumptions: list[str]


# Internal output structures
class QuestionFormulationDecision(BaseModel):
    needs_clarification: bool
    clarification_question: str | None = None


# class QuestionFormulationResult(BaseModel):
#     decision: QuestionFormulationDecision


# Node definition
# PCC Question Formulation Agent TODO: Add PICOC, PICO, SPYDER, etc.
# The agent will remain the same. Only the prompts for the agent will change.
class QuestionFormulationAgent:
    def __init__(self, model_name, max_user_clarification_count, openai_key):

        self.model = init_chat_model(
            model_name, model_provider="openai", api_key=openai_key, temperature=0
        )
        self.max_user_clarification_count = max_user_clarification_count

    def clarification_node(self, state: dict):
        result = self.model.with_structured_output(QuestionFormulationDecision).invoke(
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
            "needs_clarification": result.needs_clarification,
            "clarification_question": result.clarification_question or "",
            "llm_call": state.get("llm_call", 0) + 1,
        }

    def researh_brief_generation_node(self, state: dict):
        """Generate initial research formuation in structured output"""
        brief = self.model.with_structured_output(FormulatedQuestion).invoke(
            [SystemMessage(content=render_yaml_prompt(prompt_research_brief))]
            + state["messages"]
        )

        return {"formulated_question": brief, "llm_call": state.get("llm_call", 0) + 1}

    def user_input_node(self, state: dict):
        question = state["clarification_question"]
        observation = input(f"Agent: {question}\nYour answer: ")
        return {
            "messages": [
                AIMessage(content=question),
                HumanMessage(content=observation),
            ],
            "user_clarification_count": state.get("user_clarification_count", 0) + 1,
        }

    def router_node(
        self,
        state: QuestionFormulationState,
    ) -> Literal["clarification_node", "researh_brief_generation_node"]:
        """Decide whether to take user input or generate the brief."""
        if (
            state.get("needs_clarification", False)
            and state.get("user_clarification_count", 0)
            < self.max_user_clarification_count
        ):
            return "user_input_node"
        return "researh_brief_generation_node"

    def build_graph(
        self,
    ):
        # Build workflow
        agent_builder = StateGraph(QuestionFormulationState)

        # Add nodes
        agent_builder.add_node("clarification_node", self.clarification_node)
        agent_builder.add_node(
            "researh_brief_generation_node", self.researh_brief_generation_node
        )
        agent_builder.add_node("user_input_node", self.user_input_node)

        # Add edges
        agent_builder.add_edge(START, "clarification_node")
        agent_builder.add_conditional_edges(
            "clarification_node",
            self.router_node,
            ["user_input_node", "researh_brief_generation_node"],
        )
        agent_builder.add_edge("user_input_node", "clarification_node")
        agent_builder.add_edge("researh_brief_generation_node", END)

        return agent_builder


if __name__ == "__main__":
    import os
    from pprint import pprint

    from dotenv import load_dotenv

    load_dotenv()

    question_formulation_agent = (
        QuestionFormulationAgent(
            model_name=os.environ.get("OPENAI_MODEL_NAME"),
            max_user_clarification_count=3,
            openai_key=os.environ.get("OPENAI_API_KEY"),
        )
        .build_graph()
        .compile()
    )

    query = input(f"Enter your research query: ")
    messages = [HumanMessage(content=query)]
    result = question_formulation_agent.invoke({"messages": messages})
    for m in result["messages"]:
        m.pretty_print()
    print(result.keys())
    pprint(result["formulated_question"])
