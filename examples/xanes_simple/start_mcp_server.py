import os
import sys
import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph

query = "Compute the XANES for Ti in TiO2 and Fe in Fe2O3"
config = {"thread_id": "test_session_001"}

async def main():
    client = MultiServerMCPClient(
        {
            "XANES MCP": {
                "transport": "stdio",
                "command": sys.executable,
                "args": ["-m", "chemgraph.mcp.xanes_mcp_parsl"],
            },
        }
    )

    tools = await client.get_tools()

    cg = ChemGraph(
        model_name="gemini-2.5-flash",
        workflow_type=os.getenv("WORKFLOW_TYPE", "multi_agent_xanes"),
        structured_output=True,
        return_option="state",
        tools=tools,
    )

    return await cg.run(query, config)

if __name__ == "__main__":
    result = asyncio.run(main())
    print(result)
