import os

from parsl.config import Config
from parsl.executors import HighThroughputExecutor
from parsl.launchers import SingleNodeLauncher
from parsl.providers import PBSProProvider


def _get_int_env(name: str, default: int) -> int:
    """Read an integer environment variable with a fallback default."""
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return int(value)


def _build_worker_init(run_dir: str) -> str:
    """Build the worker initialization script for Improv worker nodes."""
    init_lines = []

    conda_sh = os.getenv("CHEMGRAPH_CONDA_SH")
    conda_env = os.getenv("CHEMGRAPH_CONDA_ENV")
    extra_init = os.getenv("CHEMGRAPH_WORKER_INIT")
    omp_threads = os.getenv("CHEMGRAPH_OMP_NUM_THREADS", "1")
    tmpdir = os.getenv("CHEMGRAPH_TMPDIR", "/tmp")

    if conda_sh:
        init_lines.append(f'source "{conda_sh}"')
    if conda_env:
        init_lines.append(f"conda activate {conda_env}")

    init_lines.append(f"export OMP_NUM_THREADS={omp_threads}")
    init_lines.append(f"export TMPDIR={tmpdir}")
    init_lines.append(f'cd "{run_dir}"')

    if extra_init:
        init_lines.append(extra_init)

    return "\n".join(init_lines)


def _build_scheduler_options() -> str:
    """Build optional scheduler directives for the PBS worker jobs."""
    scheduler_lines = []

    queue = os.getenv("CHEMGRAPH_PBS_QUEUE")
    job_name = os.getenv("CHEMGRAPH_PBS_JOB_NAME", "ChemGraph_Improv_Parsl")
    filesystems = os.getenv("CHEMGRAPH_PBS_FILESYSTEMS")

    if job_name:
        scheduler_lines.append(f"#PBS -N {job_name}")
    if queue:
        scheduler_lines.append(f"#PBS -q {queue}")
    if filesystems:
        scheduler_lines.append(f"#PBS -l filesystems={filesystems}")

    return "\n".join(scheduler_lines)


def get_improv_config(run_dir=None):
    """Generate a Parsl configuration for Improv using PBSProProvider.

    The configuration is intentionally environment-driven so user- and
    allocation-specific values stay in the PBS launcher script instead of the
    repository code.
    """
    if run_dir is None:
        run_dir = os.getcwd()

    account = os.getenv("CHEMGRAPH_PBS_ACCOUNT")
    if not account:
        raise ValueError(
            "CHEMGRAPH_PBS_ACCOUNT is not set. Export it in the PBS script "
            "before launching ChemGraph on Improv."
        )

    walltime = os.getenv("CHEMGRAPH_PBS_WALLTIME", "01:00:00")
    cpus_per_node = _get_int_env(
        "CHEMGRAPH_CPUS_PER_NODE",
        _get_int_env("PBS_NP", 128),
    )
    max_blocks = _get_int_env("CHEMGRAPH_MAX_BLOCKS", 1)
    init_blocks = min(_get_int_env("CHEMGRAPH_INIT_BLOCKS", 1), max_blocks)
    min_blocks = _get_int_env("CHEMGRAPH_MIN_BLOCKS", 0)
    worker_debug = os.getenv("CHEMGRAPH_WORKER_DEBUG", "").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    provider = PBSProProvider(
        account=account,
        nodes_per_block=1,
        cpus_per_node=cpus_per_node,
        init_blocks=init_blocks,
        min_blocks=min_blocks,
        max_blocks=max_blocks,
        walltime=walltime,
        scheduler_options=_build_scheduler_options(),
        worker_init=_build_worker_init(run_dir=run_dir),
        launcher=SingleNodeLauncher(),
    )

    return Config(
        executors=[
            HighThroughputExecutor(
                label="htex",
                max_workers_per_node=1,
                cores_per_worker=cpus_per_node,
                worker_debug=worker_debug,
                provider=provider,
            )
        ],
        run_dir=run_dir,
    )
