import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from research_agent.state import EvidenceSynthesisState, PaperProfile
from research_agent.utils import render_yaml_prompt, convert_pdf_url_to_markdown

# Load env
load_dotenv()

# Define prompts
prompt_claims_extraction = "/Users/deepankersingh/Projects/research-agent/src/research_agent/prompts/claims_extraction.yaml"


# Node definition
class EvidenceSynthesisAgent:
    def __init__(self, model_name, openai_key):

        self.model = init_chat_model(
            model_name, model_provider="openai", api_key=openai_key, temperature=0
        )

    def _extract_paper_data(self, paper_data):
        extracted_paper_data = convert_pdf_url_to_markdown(paper_data.get("pdf_url"), openalex_api_key=os.environ.get("OPEN_ALEX_API_KEY"))
        
        return extracted_paper_data

    def claims_extraction_node(self, state: dict):
        filtered_retrievals = state["filtered_retrievals"]
        paper_profiles = []

        for paper_data in filtered_retrievals:

            extracted_paper_data = self._extract_paper_data(paper_data)
            paper_data["extracted_paper_data"] = extracted_paper_data

            result = self.model.with_structured_output(PaperProfile).invoke(
                [
                    SystemMessage(
                        content=render_yaml_prompt(
                            prompt_claims_extraction,
                            extracted_paper_data=extracted_paper_data,
                        )
                    )
                ]
            )
            paper_profiles.append(result.model_dump(mode="json"))

        return {
            "paper_profiles": paper_profiles,
            "llm_call": state.get("llm_call", 0) + 1,
        }

    def build_graph(
        self,
    ):
        # Build workflow
        agent_builder = StateGraph(EvidenceSynthesisState)

        # Add nodes
        agent_builder.add_node("claims_extraction_node", self.claims_extraction_node)

        # Add edges
        agent_builder.add_edge(START, "claims_extraction_node")
        agent_builder.add_edge("claims_extraction_node", END)

        return agent_builder


