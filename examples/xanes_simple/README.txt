#!/bin/sh
#PBS -l select=1:ncpus=128:mpiprocs=128
#PBS -A Hetero_Catalysis
#PBS -l walltime=06:00:00

cd $PBS_O_WORKDIR

export NUMEXPR_NUM_THREADS=64

export MP_API_KEY="" # MP key if fetching structure
export FDMNES_EXE="~/parallel_fdmnes/mpirun_fdmnes" # Path to FDMNES exe
export GEMINI_API_KEY="" # Or your favored LLM

source "$HOME/miniconda/etc/profile.d/conda.sh"
conda activate chemgraph
export PYTHONPATH="/lcrc/globalscratch/vferreiragrizzi/agents/ChemGraph_xanes/src:$PYTHONPATH"

export COMPUTE_SYSTEM=improv
export CHEMGRAPH_PBS_ACCOUNT="Hetero_Catalysis"
export CHEMGRAPH_PBS_WALLTIME="04:00:00"
export CHEMGRAPH_MAX_BLOCKS=2
export CHEMGRAPH_CPUS_PER_NODE=128
export CHEMGRAPH_CONDA_SH="$HOME/miniconda/etc/profile.d/conda.sh"
export CHEMGRAPH_CONDA_ENV="chemgraph"
export CHEMGRAPH_OMP_NUM_THREADS=1

export WORKFLOW_TYPE="multi_agent_xanes"
python /home/vferreiragrizzi/scripts/chemgraph/run_xanes_agent.py