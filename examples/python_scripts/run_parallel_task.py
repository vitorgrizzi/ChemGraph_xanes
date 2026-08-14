import asyncio
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from langchain_mcp_adapters.client import MultiServerMCPClient
from chemgraph.agent.llm_agent import ChemGraph


MODEL_NAME = "gemini-2.5-flash"
WORKFLOW_TYPE = "multi_agent_xanes"
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

EXPERIMENT_ROOT = Path(
    os.getenv("XANES_EXPERIMENT_ROOT", Path.cwd())
).resolve()

EXPERIMENT_DIR = EXPERIMENT_ROOT / f"tio2_batch_{RUN_ID}"
RUNS_DIR = EXPERIMENT_DIR / "fdmnes_runs"
LOG_DIR = EXPERIMENT_DIR / "agent_logs"
PARSL_DIR = EXPERIMENT_DIR / "parsl"

EXPERIMENT_DIR.mkdir(parents=True, exist_ok=False)
LOG_DIR.mkdir(parents=True)
PARSL_DIR.mkdir(parents=True)

# Ensure the selected workflow and all generated logs are experiment-specific.
os.environ["WORKFLOW_TYPE"] = WORKFLOW_TYPE
os.environ["CHEMGRAPH_LOG_DIR"] = str(LOG_DIR)
os.environ["CHEMGRAPH_PARSL_RUN_DIR"] = str(PARSL_DIR)

required_variables = [
    "GEMINI_API_KEY",
    "MP_API_KEY",
    "FDMNES_EXE",
    "COMPUTE_SYSTEM",
    "CHEMGRAPH_PBS_ACCOUNT",
]

missing = [name for name in required_variables if not os.getenv(name)]
if missing:
    raise RuntimeError(
        "Missing required environment variables: " + ", ".join(missing)
    )

if os.environ["COMPUTE_SYSTEM"].lower() != "improv":
    raise RuntimeError(
        "This launcher expects COMPUTE_SYSTEM=improv, but received "
        f"{os.environ['COMPUTE_SYSTEM']!r}."
    )


QUERY = f"""
Fetch all non-deprecated TiO2 structures from Materials Project with
energy_above_hull <= 0.10 eV/atom. Pass exactly the returned structure_files
list to one run_xanes_ensemble call; do not scan the output directory or call
run_xanes_single.

Run Ti K-edge calculations with:
output_dir="{RUNS_DIR}", z_absorber=22, edge="K", radius=5.0,
energy_range=[-20.0, 1.0, -10.0, 0.05, 10.0, 0.1, 50.0],
green=true, density_all=false, quadrupole=true, spherical=false,
scf=true, magnetism=false, and skip_completed=false.

Do not plot or modify the spectra. Report the retrieved, successful, and failed
structure counts and the absolute path to xanes_results.jsonl.
""".strip()


def write_run_metadata():
    """Record non-secret execution settings before starting the agent."""

    metadata = {
        "run_id": RUN_ID,
        "start_time_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": MODEL_NAME,
        "workflow_type": WORKFLOW_TYPE,
        "experiment_directory": str(EXPERIMENT_DIR),
        "runs_directory": str(RUNS_DIR),
        "parsl_directory": str(PARSL_DIR),
        "compute_system": os.environ["COMPUTE_SYSTEM"],
        "fdmnes_executable": os.environ["FDMNES_EXE"],
        "pbs_account": os.environ["CHEMGRAPH_PBS_ACCOUNT"],
        "pbs_worker_walltime": os.getenv("CHEMGRAPH_PBS_WALLTIME"),
        "maximum_parsl_blocks": os.getenv("CHEMGRAPH_MAX_BLOCKS"),
        "cpus_per_worker_node": os.getenv("CHEMGRAPH_CPUS_PER_NODE"),
        "omp_threads": os.getenv("CHEMGRAPH_OMP_NUM_THREADS"),
        "query": QUERY,
    }

    metadata_file = EXPERIMENT_DIR / "run_metadata.json"
    with metadata_file.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    query_file = EXPERIMENT_DIR / "agent_query.txt"
    query_file.write_text(QUERY + "\n", encoding="utf-8")


async def main():
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
    print(f"Experiment directory: {EXPERIMENT_DIR}")
    print(f"Model: {MODEL_NAME}")
    print(f"Workflow: {WORKFLOW_TYPE}")

    cg = ChemGraph(
        model_name=MODEL_NAME,
        workflow_type=WORKFLOW_TYPE,
        structured_output=True,
        return_option="state",
        tools=tools,
        log_dir=str(LOG_DIR),
    )

    config = {
        "thread_id": f"tio2_batch_{RUN_ID}",
    }

    return await cg.run(QUERY, config)


if __name__ == "__main__":
    write_run_metadata()

    try:
        result = asyncio.run(main())

        state_file = EXPERIMENT_DIR / "agent_state.json"
        with state_file.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)

        print(f"Experiment completed: {EXPERIMENT_DIR}")
        print(f"Agent state: {state_file}")
        print(result)

    except Exception:
        failure_file = EXPERIMENT_DIR / "run_failure.txt"
        failure_file.write_text(traceback.format_exc(), encoding="utf-8")

        print(f"Experiment failed. Traceback saved to: {failure_file}")
        raise
