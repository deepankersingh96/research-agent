from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from typing import Literal
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel, Field

from research_agent.state import RetrieverState, ListRetrievals
from research_agent.utils import render_yaml_prompt
from research_agent.tools.retrieval_tools import search_arxiv

# Define prompts


# Node definition
class RetrievalAgent:
    def __init__(self, model_name, openai_key):

        self.model = init_chat_model(
            model_name, model_provider="openai", api_key=openai_key, temperature=0
        )
        self.tools = [search_arxiv]
        self.tools_by_names = {tool.name: tool for tool in self.tools}
        self.model_with_tools = self.model.bind_tools(self.tools)

    def retriever_node(self, state: dict):
        print("Running Retriever Node")
        result = self.model_with_tools.invoke(
            [
                SystemMessage(
                    content=f"""
You are a retrieval agent who's job is to search research papers in dabases like arxiv, semantic scholar, pubmed, etc. 
You must use only the available tools for a specified database to retrieve results against the query strings. 

The expanded search queries are given below:

{{state["expanded_queries"]["queries"]}}
"""
                )
            ]
        )
        return {
            "messages": [result],
            "llm_call": state.get("llm_call", 0) + 1,
        }

    def tool_node(self, state: dict):
        print("Running Tool Node")
        results = []
        for tool_call in state["messages"][-1].tool_calls:
            tool = self.tools_by_names[tool_call["name"]]
            observation = tool.invoke(tool_call["args"])
            results.append(
                ToolMessage(content=observation, tool_call_id=tool_call["id"])
            )
        return {
            "messages": results,
        }

    def filteration_node(self, state: dict):
        print("Running filteration Node")
        """Filter results that are relevant for the user query using LLM judge"""
        retrievals = state["messages"][-1].content  # list of retrievals
        print(retrievals)

        all_filtered_retrievals = []

        for ret in retrievals:
            filtered_retreivals = self.model.with_structured_output(
                ListRetrievals
            ).invoke(
                [SystemMessage(content=f"""
You are a research assistant agent, your job is to filter the retreived papers from databases like arxiv, pubmed, etc. based on the user query. 
You will be provided the original research question formulated by the user. 
Your main job is to check the abstract or summary of the fetched papers and select only the relevant papers from the entire list. 
Give your output in the defined structured output format. 

Original user question: 
{{state.formulated_question}}


Retreived papers:
{{ret}}
""")]
            )

            all_filtered_retrievals.extend(filtered_retreivals.content)

        return {"filtered_retrievals": all_filtered_retrievals}

    def deduplication_node(self, state: dict):
        print("Running deduplication node")
        filtered_retrievals = state["filtered_retrievals"]
        deduplicated_retrievals: ListRetrievals = []
        ids = set()

        for ret in filtered_retrievals:
            if ret["short_id"] in ids:
                continue
            deduplicated_retrievals.append(ret)

        return {"deduplicated_retrievals": deduplicated_retrievals}

    def build_graph(
        self,
    ):
        # Build workflow
        agent_builder = StateGraph(RetrieverState)

        # Add nodes
        agent_builder.add_node("retriever_node", self.retriever_node)
        agent_builder.add_node("tool_node", self.tool_node)
        agent_builder.add_node("filteration_node", self.filteration_node)
        agent_builder.add_node("deduplication_node", self.deduplication_node)

        # Add edges
        agent_builder.add_edge(START, "retriever_node")
        agent_builder.add_edge("retriever_node", "tool_node")
        agent_builder.add_edge("tool_node", "filteration_node")
        agent_builder.add_edge("filteration_node", "deduplication_node")
        agent_builder.add_edge("deduplication_node", END)

        return agent_builder


if __name__ == "__main__":
    import os
    import json
    from pprint import pprint

    from dotenv import load_dotenv

    load_dotenv()

    retreival_agent = (
        RetrievalAgent(
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

    # Load query expansion stub.
    with open(
        "/Users/deepankersingh/Projects/research-agent/test/stubs/output_query_expansion.json",
        "r",
    ) as fp:
        output_query_expansion = json.load(fp)

    # Prepare input state.
    result = retreival_agent.invoke(
        {
            "formulated_question": output_question_formulation,
            "expanded_queries": output_query_expansion,
        }
    )
    # Print results.
    for m in result["messages"]:
        m.pretty_print()
    print(result.keys())
    pprint(result["deduplicated_retrievals"])

    # Save results.
    with open(
        "/Users/deepankersingh/Projects/research-agent/test/stubs/output_retriever.json",
        "w",
    ) as fp:
        json.dump(result.model_dump(mode="json"), fp, indent=4)
