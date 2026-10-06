import os
from dotenv import load_dotenv

from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

from research_agent.state import (
    EvidenceSynthesisState,
    EvidenceClaim,
    PaperProfile,
    PaperProfileExtraction,
    TrendAnalysis,
    GapAnalysis,
    ContradictionAnalysis,
)
from research_agent.utils import render_yaml_prompt, convert_pdf_url_to_markdown

# Load env
load_dotenv()

# Define prompts
prompt_claims_extraction = "/Users/deepankersingh/Projects/research-agent/src/research_agent/prompts/claims_extraction.yaml"
prompt_trend_analysis = "/Users/deepankersingh/Projects/research-agent/src/research_agent/prompts/trend_analysis.yaml"
prompt_gap_identification = "/Users/deepankersingh/Projects/research-agent/src/research_agent/prompts/gap_identification.yaml"
prompt_contradiction_detection = "/Users/deepankersingh/Projects/research-agent/src/research_agent/prompts/contradiction_detection.yaml"


# Node definition
class EvidenceSynthesisAgent:
    def __init__(self, model_name, openai_key):

        self.model = init_chat_model(
            model_name, model_provider="openai", api_key=openai_key, temperature=0
        )

    def _extract_paper_data(self, paper_data):
        extracted_paper_data = convert_pdf_url_to_markdown(
            paper_data.get("pdf_url"),
            openalex_api_key=os.environ.get("OPEN_ALEX_API_KEY"),
        )

        return extracted_paper_data

    def claims_extraction_node(self, state: dict):
        filtered_retrievals = state["filtered_retrievals"]
        paper_profiles = []

        for paper_data in filtered_retrievals:

            extracted_paper_data = self._extract_paper_data(paper_data)
            paper_data["extracted_paper_data"] = extracted_paper_data
            paper_id = paper_data["short_id"]

            extracted_paper_profile = self.model.with_structured_output(PaperProfileExtraction).invoke(
                [
                    SystemMessage(
                        content=render_yaml_prompt(
                            prompt_claims_extraction,
                            extracted_paper_data=extracted_paper_data,
                        )
                    )
                ]
            )

            # Add unique ID to each emperical claim
            claims = [
                EvidenceClaim(
                    claim_id=f"{paper_id}_C{i}",
                    **claim.model_dump()
                )
                for i, claim in enumerate( extracted_paper_profile.empirical_claims, start=1)
            ]

            # Add unique ID to each paper profile
            paper_profile = PaperProfile(
                paper_id=paper_id, 
                empirical_claims=claims,
                **extracted_paper_profile.model_dump(exclude={"empirical_claims"})
            )

            paper_profiles.append(paper_profile.model_dump(mode="json"))

        return {
            "paper_profiles": paper_profiles,
            "llm_call": len(filtered_retrievals),
        }

    def trend_analysis_node(self, state: dict):
        paper_profiles = state["paper_profiles"]

        result = self.model.with_structured_output(TrendAnalysis).invoke(
            [
                SystemMessage(
                    content=render_yaml_prompt(
                        prompt_trend_analysis,
                        formulated_question=state["formulated_question"],
                        extracted_claims=[
                            profile for profile in paper_profiles if profile["empirical_claims"]
                        ],
                    )
                )
            ]
        )

        return {"llm_call": 1, "trends": result.model_dump()}

    def gap_identification_node(self, state: dict):
        paper_profiles = state["paper_profiles"]

        result = self.model.with_structured_output(GapAnalysis).invoke(
            [
                SystemMessage(
                    content=render_yaml_prompt(
                        prompt_gap_identification,
                        formulated_question=state["formulated_question"],
                        extracted_claims=[
                            profile for profile in paper_profiles if profile["empirical_claims"]
                        ],
                    )
                )
            ]
        )

        return {"llm_call": 1, "gaps": result.model_dump()}

    def contradiction_detection_node(self, state: dict):
        paper_profiles = state["paper_profiles"]

        result = self.model.with_structured_output(ContradictionAnalysis).invoke(
            [
                SystemMessage(
                    content=render_yaml_prompt(
                        prompt_contradiction_detection,
                        formulated_question=state["formulated_question"],
                        extracted_claims=[
                            profile for profile in paper_profiles if profile["empirical_claims"]
                        ],
                    )
                )
            ]
        )

        return {"llm_call": 1, "contradictions": result.model_dump()}

    def build_graph(
        self,
    ):
        # Build workflow
        agent_builder = StateGraph(EvidenceSynthesisState)

        # Add nodes
        agent_builder.add_node("claims_extraction_node", self.claims_extraction_node)
        agent_builder.add_node("trend_analysis_node", self.trend_analysis_node)
        agent_builder.add_node("gap_identification_node", self.gap_identification_node)
        agent_builder.add_node("contradiction_detection_node", self.contradiction_detection_node)

        # Add edges
        agent_builder.add_edge(START, "claims_extraction_node")
        agent_builder.add_edge("claims_extraction_node", "trend_analysis_node")
        agent_builder.add_edge("claims_extraction_node", "gap_identification_node")
        agent_builder.add_edge("claims_extraction_node", "contradiction_detection_node")
        agent_builder.add_edge("trend_analysis_node", END)
        agent_builder.add_edge("gap_identification_node", END)
        agent_builder.add_edge("contradiction_detection_node", END)

        return agent_builder
