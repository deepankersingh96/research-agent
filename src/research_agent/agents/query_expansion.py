from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from research_agent.state import QueryExpansionState, ExpandedQuery
from research_agent.utils import render_yaml_prompt

# Define prompts
prompt_query_expansion = "src/research_agent/prompts/query_expansion.yaml"


# Node definition
class QueryExpansionAgent:
    def __init__(self, model_name, openai_key):

        self.model = init_chat_model(
            model_name, model_provider="openai", api_key=openai_key, temperature=0
        )

    def query_expansion_node(self, state: dict):
        result = self.model.with_structured_output(ExpandedQuery).invoke(
            [
                SystemMessage(
                    content=render_yaml_prompt(
                        prompt_query_expansion,
                        formulated_question=state["formulated_question"],
                    )
                )
            ]
        )
        return {
            "expanded_queries": result,
            "llm_call": state.get("llm_call", 0) + 1,
        }

    def build_graph(
        self,
    ):
        # Build workflow
        agent_builder = StateGraph(QueryExpansionState)

        # Add nodes
        agent_builder.add_node("query_expansion_node", self.query_expansion_node)

        # Add edges
        agent_builder.add_edge(START, "query_expansion_node")
        agent_builder.add_edge("query_expansion_node", END)

        return agent_builder


if __name__ == "__main__":
    import os
    import json
    from pprint import pprint

    from dotenv import load_dotenv

    load_dotenv()

    query_expansion_agent = (
        QueryExpansionAgent(
            model_name=os.environ.get("OPENAI_MODEL_NAME"),
            openai_key=os.environ.get("OPENAI_API_KEY"),
        )
        .build_graph()
        .compile()
    )

    # Load question formulation stub.
    with open(
        "/Users/deepankersingh/Projects/research-agent/test/stubs/output_question_formulation.json",
        "r",
    ) as fp:
        output_question_formulation = json.load(fp)

    # Prepare input state.
    result = query_expansion_agent.invoke(
        {"formulated_question": output_question_formulation}
    )
    # Print results.
    for m in result["messages"]:
        m.pretty_print()
    print(result.keys())
    pprint(result["expanded_queries"])

    # Save results. 
    with open("/Users/deepankersingh/Projects/research-agent/test/stubs/ouput_query_expansion.json", "w") as fp: 
            json.dump(result["expanded_queries"].model_dump(mode="json"), fp, indent=4)
