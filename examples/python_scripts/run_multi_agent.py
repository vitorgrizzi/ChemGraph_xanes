import asyncio
import os
import sys
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph
from chemgraph.tools.rag_tools import load_document, query_knowledge_base

job_dir = Path(os.environ["PBS_O_WORKDIR"]).resolve()
poscar_path = job_dir / "POSCAR"
tio2_output = job_dir / "TiO2_xanes"
fe2o3_output = job_dir / "Fe2O3_xanes"

# QUERY = "Compute the XANES for Ti in TiO2"
# QUERY = f"Compute the XANES of the POSCAR file located at '{poscar_path}' for Z absorber = 29"
QUERY = f"""
Load the FDMNES manual from
/lcrc/globalscratch/vferreiragrizzi/agents/success_rates/rag/FDMNES_manual.txt
using HuggingFace embeddings.

Query the manual for the default values of the FDMNES Range keyword.

Then compute the XANES for Ti in TiO2 and Fe in Fe2O3. Use the retrieved
Range values to populate the XANES tool argument energy_range. Do not guess
default numerical values.

Write the TiO2 calculation to "{tio2_output}" and the Fe2O3 calculation to
"{fe2o3_output}".
""".strip()

CONFIG = {
    "thread_id": "test_session_001",
}

MODEL_NAME = "gemini-2.5-flash"
WORKFLOW_TYPE = os.getenv("WORKFLOW_TYPE", "multi_agent_xanes")


def validate_environment() -> None:
    """Validate required settings without printing secret values."""

    required_variables = [
        "GEMINI_API_KEY",
        "MP_API_KEY",
        "FDMNES_EXE",
        "COMPUTE_SYSTEM",
        "CHEMGRAPH_PBS_ACCOUNT",
    ]

    missing = [
        variable
        for variable in required_variables
        if not os.environ.get(variable)
    ]

    if missing:
        raise RuntimeError(
            "Missing required environment variables: " + ", ".join(missing)
        )


async def main():
    validate_environment()

    # Explicit propagation is necessary because the MCP stdio transport does
    # not inherit MP_API_KEY, FDMNES_EXE, or CHEMGRAPH_* variables by default.
    mcp_environment = dict(os.environ)

    client = MultiServerMCPClient(
        {
            "XANES MCP": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [
                    "-u",
                    "-m",
                    "chemgraph.mcp.xanes_mcp_parsl",
                ],
                "env": mcp_environment,
            },
        }
    )

    tools = await client.get_tools()

    print(f"Connected tools: {[tool.name for tool in tools]}")
    print(f"Model: {MODEL_NAME}")
    print(f"Workflow: {WORKFLOW_TYPE}")
    print(f"Query: {QUERY}")

    cg = ChemGraph(
        model_name=MODEL_NAME,
        workflow_type=WORKFLOW_TYPE,
        structured_output=True,
        return_option="state",
        tools=tools,
    )

    return await cg.run(QUERY, CONFIG)


if __name__ == "__main__":
    result = asyncio.run(main())
    print(result)
