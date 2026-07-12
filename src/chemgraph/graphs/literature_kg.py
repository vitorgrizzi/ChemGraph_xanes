"""LangGraph controller for the literature KG workflow."""

from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from chemgraph.kg.agents import kg_langchain_tools
from chemgraph.state.state import State
from chemgraph.graphs.rag_agent import route_tools
from chemgraph.utils.logging_config import setup_logger

logger = setup_logger(__name__)


literature_kg_prompt = """You are ChemGraph's literature knowledge-graph agent.

Use the KG tools to ingest papers, extract CatalystRecord JSON, build an
evidence-backed graph, answer hybrid graph/RAG questions, and generate
hypothesis cards. Do not invent literature facts. Every scientific claim you
surface should point to evidence IDs or say that evidence is missing. Expensive
computational or experimental actions require human approval; propose structured
tasks instead of launching them. Graph construction performs mandatory
verification; never request an unsafe bypass. Use kg_validate_graph after a
build, use kg_verify_records when the user needs an issue report, and treat
hypothesis cards as trend candidates rather than causal proof.
"""


def LiteratureKGAgent(state: State, llm, system_prompt: str, tools=None):
    if tools is None:
        tools = kg_langchain_tools()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"{state['messages']}"},
    ]
    llm_with_tools = llm.bind_tools(tools=tools)
    return {"messages": [llm_with_tools.invoke(messages)]}


def construct_literature_kg_graph(
    llm,
    system_prompt: str = literature_kg_prompt,
    tools: list | None = None,
):
    """Construct a LangGraph ReAct controller over literature KG tools."""
    logger.info("Constructing literature KG agent graph")
    checkpointer = MemorySaver()
    tools = tools or kg_langchain_tools()
    tool_node = ToolNode(tools=tools)
    graph_builder = StateGraph(State)
    graph_builder.add_node(
        "LiteratureKGAgent",
        lambda state: LiteratureKGAgent(
            state,
            llm,
            system_prompt=system_prompt,
            tools=tools,
        ),
    )
    graph_builder.add_node("tools", tool_node)
    graph_builder.add_edge(START, "LiteratureKGAgent")
    graph_builder.add_conditional_edges(
        "LiteratureKGAgent",
        route_tools,
        {"tools": "tools", "done": END},
    )
    graph_builder.add_edge("tools", "LiteratureKGAgent")
    return graph_builder.compile(checkpointer=checkpointer)
