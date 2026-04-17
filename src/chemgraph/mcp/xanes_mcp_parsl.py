import asyncio
import json
import logging
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

import parsl
from parsl import bash_app

from chemgraph.mcp.server_utils import run_mcp_server
from chemgraph.schemas.xanes_schema import (
    xanes_input_schema,
    xanes_input_schema_ensemble,
    mp_query_schema,
)


@bash_app
def run_fdmnes_parsl_app(
    run_dir: str,
    fdmnes_exe: str,
    stdout=None,
    stderr=None,
):
    """Parsl bash_app that runs FDMNES in a prepared input directory.

    Parameters
    ----------
    run_dir : str
        Path to the directory containing fdmfile.txt and fdmnes_in.txt.
    fdmnes_exe : str
        Path to the FDMNES executable.
    """
    return f'cd "{run_dir}" && "{fdmnes_exe}"'


def load_parsl_config(system_name: str):
    """Dynamically import and return a Parsl config for the given HPC system.

    Parameters
    ----------
    system_name : str
        Target system name. Supported: ``polaris``, ``aurora``, ``improv``.
    """
    system_name = system_name.lower()
    run_dir = os.getcwd()

    logging.info("Initializing Parsl for system: %s", system_name)

    if system_name == "polaris":
        from chemgraph.hpc_configs.polaris_parsl import get_polaris_config

        return get_polaris_config(run_dir=run_dir)

    elif system_name == "aurora":
        from chemgraph.hpc_configs.aurora_parsl import get_aurora_config

        return get_aurora_config(run_dir=run_dir)

    elif system_name == "improv":
        from chemgraph.hpc_configs.improv_parsl import get_improv_config

        return get_improv_config(run_dir=run_dir)

    else:
        raise ValueError(
            "Unknown system specified: "
            f"'{system_name}'. Supported: polaris, aurora, improv"
        )


# Load Parsl config at module level (same pattern as graspa_mcp_parsl.py)
target_system = os.getenv("COMPUTE_SYSTEM", "polaris")
parsl.load(load_parsl_config(target_system))

# Start MCP server
mcp = FastMCP(
    name="ChemGraph XANES Tools",
    instructions="""
        You expose tools for running XANES/FDMNES simulations.
        The available tools are:
        1. run_xanes_single: run a single FDMNES calculation for one structure.
        2. run_xanes_ensemble: run FDMNES calculations over multiple structures
           from structure files or ASE databases using Parsl for parallel execution.
        3. fetch_mp_structures: fetch optimized structures from Materials Project.
        4. plot_xanes: generate normalized XANES plots for completed calculations.

        Guidelines:
        - Use each tool only when its input schema matches the user request.
        - Do not guess numerical values; report tool errors exactly as they occur.
        - Keep responses compact -- full results are in the output directories.
        - When returning paths, use absolute paths.
        - Energies are in eV.
    """,
)


@mcp.tool(
    name="run_xanes_single",
    description="Run a single XANES/FDMNES calculation for one input structure.",
)
def run_xanes_single(params: xanes_input_schema):
    """Run a single FDMNES calculation using the core engine."""
    from chemgraph.tools.xanes_tools import run_xanes_core

    return run_xanes_core(params)


@mcp.tool(
    name="run_xanes_ensemble",
    description="Run an ensemble of XANES/FDMNES calculations using Parsl.",
)
async def run_xanes_ensemble(params: xanes_input_schema_ensemble):
    """Run ensemble XANES calculations over multiple structures using Parsl.

    For each structure source entry:
    1. Reads the structure(s) from files or an ASE database.
    2. Creates FDMNES input files in a per-structure subdirectory.
    3. Submits a Parsl bash_app to run FDMNES.
    4. Gathers results and writes a JSONL summary log.

    Parameters
    ----------
    params : xanes_input_schema_ensemble
        Input parameters for the ensemble calculation.
    """
    from chemgraph.tools.xanes_tools import (
        extract_conv,
        prepare_xanes_batch,
    )

    batch = prepare_xanes_batch(
        input_source=params.input_structures,
        z_absorber=params.z_absorber,
        radius=params.radius,
        magnetism=params.magnetism,
        output_dir=params.output_dir,
        ase_db_selection=params.ase_db_selection,
        skip_completed=params.skip_completed,
    )

    output_dir = Path(batch["root_dir"])
    fdmnes_exe = params.fdmnes_exe
    pending_tasks = []
    results = []

    for job in batch["jobs"]:
        run_dir = Path(job["run_dir"])
        if job["status"] == "skipped_existing":
            try:
                conv_data = extract_conv(run_dir)
                results.append(
                    {
                        **job,
                        "status": "success",
                        "n_conv_files": len(conv_data),
                        "message": "Skipped execution because outputs already exist.",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        **job,
                        "status": "failure",
                        "error_type": type(e).__name__,
                        "message": str(e),
                    }
                )
            continue

        fut = run_fdmnes_parsl_app(
            run_dir=str(run_dir),
            fdmnes_exe=fdmnes_exe,
            stdout=str(run_dir / "fdmnes_stdout.txt"),
            stderr=str(run_dir / "fdmnes_stderr.txt"),
        )
        pending_tasks.append((job, fut))

    async def wait_for_task(meta, parsl_future):
        try:
            await asyncio.wrap_future(parsl_future)
            conv_data = extract_conv(meta["run_dir"])
            return {
                **meta,
                "status": "success",
                "n_conv_files": len(conv_data),
            }
        except Exception as e:
            return {
                **meta,
                "status": "failure",
                "error_type": type(e).__name__,
                "message": str(e),
            }

    if pending_tasks:
        results.extend(
            await asyncio.gather(
                *(wait_for_task(meta, fut) for meta, fut in pending_tasks)
            )
        )

    summary_log_path = output_dir / "xanes_results.jsonl"
    success_count = 0

    with open(summary_log_path, "a", encoding="utf-8") as f:
        for res in results:
            if res.get("status") == "success":
                success_count += 1
            f.write(json.dumps(res) + "\n")

    return (
        f"Ensemble execution completed. Ran {len(results)} tasks "
        f"({success_count} successful, {batch['n_skipped']} reused existing outputs). "
        f"Detailed results appended to '{summary_log_path}'."
    )


@mcp.tool(
    name="fetch_mp_structures",
    description="Fetch optimized structures from Materials Project.",
)
def fetch_mp_structures(params: mp_query_schema):
    """Fetch structures from Materials Project and save as CIF files and pickle database."""
    from chemgraph.tools.xanes_tools import (
        fetch_materials_project_data,
        _get_data_dir,
    )

    data_dir = _get_data_dir()
    result = fetch_materials_project_data(params, data_dir)
    return {
        "status": "success",
        "n_structures": result["n_structures"],
        "chemsys": params.chemsys,
        "output_dir": str(data_dir),
        "structure_files": result["structure_files"],
        "pickle_file": result["pickle_file"],
    }


@mcp.tool(
    name="plot_xanes",
    description="Generate normalized XANES plots for completed FDMNES calculations.",
)
def plot_xanes(runs_dir: str):
    """Generate XANES plots for all completed runs in a directory.

    Parameters
    ----------
    runs_dir : str
        Path to the ``fdmnes_batch_runs`` directory containing ``run_*``
        subdirectories with FDMNES outputs.
    """
    from chemgraph.tools.xanes_tools import (
        plot_xanes_results,
        _get_data_dir,
    )

    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        raise ValueError(f"'{runs_dir}' is not a valid directory.")

    data_dir = _get_data_dir()
    result = plot_xanes_results(data_dir, runs_path)
    return {
        "status": "success",
        "n_plots": result["n_plots"],
        "n_failed": result["n_failed"],
        "plot_files": result["plot_files"],
        "failed": result["failed"],
    }


if __name__ == "__main__":
    run_mcp_server(mcp, default_port=9007)
