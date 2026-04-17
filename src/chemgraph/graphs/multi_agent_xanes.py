from typing import Any, List

from langchain_openai import ChatOpenAI

from chemgraph.graphs.multi_agent_mcp import construct_multi_agent_mcp_graph
from chemgraph.tools.cheminformatics_tools import (
    molecule_name_to_smiles,
    smiles_to_coordinate_file,
)
from chemgraph.tools.xanes_tools import (
    fetch_xanes_data,
    plot_xanes_data,
    run_xanes,
)
from chemgraph.prompt.xanes_prompt import (
    xanes_planner_prompt,
    xanes_executor_prompt,
    xanes_aggregator_prompt,
    xanes_formatter_prompt,
)


def construct_multi_agent_xanes_graph(
    llm: ChatOpenAI,
    planner_prompt: str = xanes_planner_prompt,
    executor_prompt: str = xanes_executor_prompt,
    aggregator_prompt: str = xanes_aggregator_prompt,
    formatter_prompt: str = xanes_formatter_prompt,
    structured_output: bool = False,
    tools: List[Any] = None,
    support_structured_output: bool = True,
):
    """Construct a XANES-focused multi-agent workflow.

    This reuses the generic multi-agent MCP graph engine but supplies
    XANES-specific prompts and accepts externally loaded tools, including
    MCP tools served by the Parsl-backed XANES server.
    """
    if tools is None:
        tools = [
            run_xanes,
            fetch_xanes_data,
            plot_xanes_data,
            molecule_name_to_smiles,
            smiles_to_coordinate_file,
        ]

    return construct_multi_agent_mcp_graph(
        llm=llm,
        planner_prompt=planner_prompt,
        executor_prompt=executor_prompt,
        aggregator_prompt=aggregator_prompt,
        formatter_prompt=formatter_prompt,
        structured_output=structured_output,
        tools=tools,
        support_structured_output=support_structured_output,
    )
