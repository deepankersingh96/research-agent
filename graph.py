from __future__ import annotations

import os

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from state import AgentState
from subgraphs.question_formulation import QuestionFormulation
from subgraphs.query_expansion import QueryExpansion

def build_agent():
    load_dotenv()

    model_name = os.environ.get("OPENAI_MODEL_NAME")
    openai_key = os.environ.get("OPENAI_API_KEY")
    max_user_clarification_count = 3

    question_formulation_subgraph = QuestionFormulation(
        model_name=model_name,
        max_user_clarification_count=max_user_clarification_count,
        openai_key=openai_key,
    ).build_question_formulation_subgraph().compile()

    query_expansion_subgraph = QueryExpansion(
        model_name=model_name,
        openai_key=openai_key,
    ).build_query_expansion_subgraph().compile()

    builder = StateGraph(AgentState)

    builder.add_node("question_formulation", question_formulation_subgraph)
    builder.add_node("query_expansion", query_expansion_subgraph)

    builder.add_edge(START, "question_formulation")
    builder.add_edge("question_formulation", "query_expansion")
    builder.add_edge("query_expansion", END)

    return builder.compile()


agent = build_agent()