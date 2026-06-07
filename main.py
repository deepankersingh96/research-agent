from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_core.messages import AnyMessage, ToolMessage, SystemMessage, HumanMessage
from typing_extensions import TypedDict, Annotated
from dotenv import load_dotenv

import operator
import os 

# Load dotenv
load_dotenv()
model_name = os.environ.get("OPENAI_MODEL_NAME")
openai_key = os.environ.get("OPENAI_API_KEY")

# Define model and tools
model = init_chat_model(
    model_name,
    model_provider="openai",
    api_key=openai_key,
    temperature=0
)

@tool
def ask_user_for_info(question: str) -> str:
    """Ask the human to provide more information to resolve ambiguities
    from the research question."""
    user_input = input(f"Agent: {question}\n Your answer:")
    return user_input 

# Augment the model with tools
tools = [ask_user_for_info]
tools_by_name = {tool.name: tool for tool in tools}
model_with_tools = model.bind_tools(tools)
    

# Define state
class ResearchState(TypedDict):
    user_query: str
    research_question_messages: Annotated[list[AnyMessage], operator.add]
    research_question: str
    llm_call: int

# Define model node
def llm_call(state: dict):
    """Formulate a research question with scope, constraint and success criteria."""

    return {
        "research_question_messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="""You are a reseach agent. 
                        This is the first step of the system. The user inputs a query
                        which is the research intent. This query can be generic or 
                        underspecified. If you find the query to be underspecified then
                        your role is to formulate a specific research question
                        from the use query. Ask user any clarification if needed. 

                        Input:
                            User query

                        Process:
                            Identify ambiguity
                            Identify missing constraints
                            Ask clarification if needed
                            Produce research question

                        Output:
                            Research Question
                            Scope
                            Constraints
                            Success Criteria"""
                    )
                ]
                + state["research_question_messages"]
            )
        ],
        "llm_calls": state.get('llm_calls', 0) + 1
    }

# Define tool node
def tool_node(state:dict):
    "Perform the tool call"

    result = []
    for tool_call in state["research_question_messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        observation = tool.invoke(tool_call["args"])
        result.append(ToolMessage(content=observation, 
                                  tool_call_id=tool_call["id"]))
    return {"research_question_messages": result}


# Define end logic
from typing import Literal
from langgraph.graph import StateGraph, START, END

def should_continue(state: ResearchState) -> Literal["tool_node", END]:
    """Decide if we should continue the loop or stop based upon whether the LLM made a tool call"""

    messages = state["research_question_messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, then perform an action
    if last_message.tool_calls:
        return "tool_node"

    # Otherwise, we stop (reply to the user)
    return END


# Build workflow
agent_builder = StateGraph(ResearchState)

# Add nodes
agent_builder.add_node("llm_call", llm_call)
agent_builder.add_node("tool_node", tool_node)

# Add edges
agent_builder.add_edge(START, "llm_call")
agent_builder.add_conditional_edges(
    "llm_call", 
    should_continue, 
    ["tool_node", END]
)
agent_builder.add_edge("tool_node", "llm_call")

# Compile the agent
agent = agent_builder.compile()


def main():
    query = input(f"Enter your research query: ")
    messages = [HumanMessage(content=query)]
    messages = agent.invoke({"research_question_messages": messages})
    for m in messages["research_question_messages"]:
        m.pretty_print()


if __name__ == "__main__":
    main()
