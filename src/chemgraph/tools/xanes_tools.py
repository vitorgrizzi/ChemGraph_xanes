import json
import logging
import os
import pickle
import subprocess
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
from ase import Atoms
from ase.io import read as ase_read, write as ase_write
from langchain_core.tools import tool

from chemgraph.schemas.xanes_schema import (
    xanes_input_schema,
    xanes_param_resolution_schema,
    mp_query_schema,
)

logger = logging.getLogger(__name__)

SUPPORTED_XANES_STRUCTURE_SUFFIXES = {".cif", ".xyz", ".poscar", ".vasp"}
CHEMGRAPH_DEFAULT_ENERGY_RANGE = [-55.0, 1.0, -10.0, 0.01, 5.0, 0.1, 150.0]
XANES_RESOLVABLE_PARAM_DEFAULTS = {
    "z_absorber": None,
    "absorber_idx": None,
    "radius": 6.0,
    "energy_range": CHEMGRAPH_DEFAULT_ENERGY_RANGE,
    "magnetism": False,
    "edge": "K",
    "green": True,
    "density_all": True,
    "quadrupole": True,
    "spherical": True,
    "scf": True,
}
XANES_PARAM_ALIASES = {
    "range": "energy_range",
    "fdmnes_range": "energy_range",
    "z": "z_absorber",
    "z_abs": "z_absorber",
    "z_absorber": "z_absorber",
    "absorber": "absorber_idx",
    "absorber_index": "absorber_idx",
    "absorber_idx": "absorber_idx",
    "radius": "radius",
    "edge": "edge",
    "magnetism": "magnetism",
    "green": "green",
    "density_all": "density_all",
    "densityall": "density_all",
    "quadrupole": "quadrupole",
    "spherical": "spherical",
    "scf": "scf",
}
XANES_PASSTHROUGH_PARAM_KEYS = {
    "input_structure_file",
    "input_source",
    "input_structure_files",
    "output_dir",
    "ase_db_selection",
    "skip_completed",
    "fdmnes_exe",
}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


def _canonical_xanes_param_key(key: str) -> str:
    """Return the canonical parameter name for common FDMNES aliases."""
    normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
    return XANES_PARAM_ALIASES.get(normalized, normalized)


def _normalize_xanes_param_map(params: dict[str, Any]) -> dict[str, Any]:
    """Normalize incoming parameter keys while dropping empty values."""
    normalized = {}
    for key, value in (params or {}).items():
        if value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        normalized[_canonical_xanes_param_key(str(key))] = value
    return normalized


def _coerce_bool(value: Any) -> bool:
    """Coerce common LLM/tool representations to a boolean."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        value_lower = value.strip().lower()
        if value_lower in {"true", "yes", "y", "1", "on", "enable", "enabled"}:
            return True
        if value_lower in {"false", "no", "n", "0", "off", "disable", "disabled"}:
            return False
    raise ValueError(f"Cannot interpret {value!r} as a boolean.")


def _coerce_xanes_param_value(key: str, value: Any) -> Any:
    """Coerce and validate a resolved XANES parameter value."""
    if key in {"z_absorber", "absorber_idx"}:
        return None if value is None else int(value)
    if key == "radius":
        return float(value)
    if key == "energy_range":
        if value is None:
            return None
        if not isinstance(value, list):
            raise ValueError("energy_range must be a list of numbers.")
        coerced = [float(v) for v in value]
        from chemgraph.schemas.xanes_schema import _validate_energy_range

        return _validate_energy_range(coerced)
    if key in {"magnetism", "green", "density_all", "quadrupole", "spherical", "scf"}:
        return _coerce_bool(value)
    if key == "edge":
        return str(value).strip().upper()
    if key in XANES_PASSTHROUGH_PARAM_KEYS:
        return value
    return value


@tool(args_schema=xanes_param_resolution_schema)
def resolve_xanes_params(
    explicit_params: dict[str, Any] | None = None,
    retrieved_params: dict[str, Any] | None = None,
    chemgraph_defaults: dict[str, Any] | None = None,
    parameters_to_resolve: list[str] | None = None,
    require_retrieval_for: list[str] | None = None,
    documentation_mode: bool = True,
    allow_default_fallback: bool = True,
) -> dict:
    """Resolve XANES/FDMNES parameters with explicit > retrieved > default priority.

    This tool is intended for documentation-grounded XANES workflows. The agent
    should first retrieve relevant FDMNES manual passages for missing parameters,
    then call this resolver with explicit user values and retrieved values. The
    returned ``final_params`` can be passed to ``run_xanes`` or to MCP wrappers
    such as ``run_xanes_single`` / ``run_xanes_ensemble``.
    """
    explicit = _normalize_xanes_param_map(explicit_params or {})
    retrieved = _normalize_xanes_param_map(retrieved_params or {})
    overrides = _normalize_xanes_param_map(chemgraph_defaults or {})

    default_values = dict(XANES_RESOLVABLE_PARAM_DEFAULTS)
    for key, value in overrides.items():
        canonical = _canonical_xanes_param_key(key)
        if canonical in default_values:
            default_values[canonical] = value

    requested = [
        _canonical_xanes_param_key(key) for key in (parameters_to_resolve or [])
    ]
    if not requested:
        requested = list(default_values.keys())

    require_retrieval_for = {
        _canonical_xanes_param_key(key) for key in (require_retrieval_for or [])
    }

    final_params = {}
    provenance = {}
    warnings = []
    missing_retrieval = []
    ready_for_xanes = True

    passthrough_keys = [
        key for key in XANES_PASSTHROUGH_PARAM_KEYS if key in explicit
    ]

    for key in requested + passthrough_keys:
        if key in explicit:
            raw_value = explicit[key]
            source = "explicit"
            reason = "User/task-specified value has highest priority."
        elif key in retrieved:
            raw_value = retrieved[key]
            source = "retrieved"
            reason = "Documentation-grounded retrieved value."
        elif key in default_values:
            raw_value = default_values[key]
            source = "chemgraph_default"
            reason = "Fallback to ChemGraph implementation default."
            if key in require_retrieval_for and documentation_mode:
                missing_retrieval.append(key)
                warnings.append(
                    f"{key} was requested for documentation retrieval but no "
                    "retrieved value was provided; using ChemGraph default."
                )
            if not allow_default_fallback:
                ready_for_xanes = False
        else:
            warnings.append(f"No value or default is available for {key}.")
            ready_for_xanes = False
            continue

        try:
            value = _coerce_xanes_param_value(key, raw_value)
        except Exception as exc:
            warnings.append(f"Invalid value for {key}: {exc}")
            ready_for_xanes = False
            continue

        if value is not None:
            final_params[key] = value
        provenance[key] = {
            "value": value,
            "source": source,
            "reason": reason,
        }

    return {
        "status": "success" if ready_for_xanes else "needs_review",
        "ready_for_xanes": ready_for_xanes,
        "policy": [
            "explicit user/task values",
            "documentation-grounded retrieved values",
            "ChemGraph implementation defaults",
        ],
        "final_params": final_params,
        "provenance": provenance,
        "missing_retrieval": sorted(missing_retrieval),
        "warnings": warnings,
    }


def write_fdmnes_input(
    ase_atoms: Atoms,
    z_absorber: int = None,
    absorber_idx: Optional[int] = None,
    input_file_dir: Path = None,
    radius: float = 6.0,
    energy_range: Optional[List[float]] = None,
    magnetism: bool = False,
    edge: str = "K",
    green: bool = True,
    density_all: bool = True,
    quadrupole: bool = True,
    spherical: bool = True,
    scf: bool = True,
):
    """Write FDMNES input files (fdmfile.txt and fdmnes_in.txt) for a structure.

    Parameters
    ----------
    ase_atoms : ase.Atoms
        Atomic structure to compute XANES for.
    z_absorber : int, optional
        Atomic number of the X-ray absorbing atom.
        Defaults to the heaviest element in the structure.
    absorber_idx : int, optional
        1-based index of the absorbing atom in the structure.
        If provided, the Absorber keyword is used instead of Z_absorber.
    input_file_dir : Path, optional
        Directory to write input files into. Defaults to cwd.
    radius : float
        Cluster radius in Angstrom. Default 6.0.
    energy_range : list of float, optional
        Values written under the FDMNES Range keyword. If omitted, uses the
        built-in ChemGraph XANES mesh.
    magnetism : bool
        Enable magnetic contributions. Default False.
    edge : str
        Absorption edge (e.g., K, L1, L2, L3, M1). Default 'K'.
    green : bool
        Use Green function method for calculation. Default True.
    density_all : bool
        Compute full electron densities. Default True.
    quadrupole : bool
        Include quadrupole transition transitions. Default True.
    spherical : bool
        Assume spherical atoms. Default True.
    scf : bool
        Perform self-consistent field calculation. Default True.
    """
    if not isinstance(ase_atoms, Atoms):
        raise TypeError("ase_atoms must be an ase.Atoms object")

    atomic_numbers = ase_atoms.get_atomic_numbers()
    if z_absorber is None:
        z_absorber = int(atomic_numbers.max())

    if input_file_dir is None:
        input_file_dir = Path.cwd()

    with open(input_file_dir / "fdmfile.txt", "w") as f:
        f.write("1\n")
        f.write("fdmnes_in.txt\n")

    with open(input_file_dir / "fdmnes_in.txt", "w") as f:
        f.write("Filout\n")
        f.write(f"{input_file_dir.name}\n\n")

        # Energy mesh
        f.write("Range\n")
        range_values = energy_range or CHEMGRAPH_DEFAULT_ENERGY_RANGE
        f.write(" ".join(f"{value:g}" for value in range_values) + "\n\n")

        # Cluster radius
        f.write("Radius\n")
        f.write(f"{radius}\n\n")

        # Absorption edge
        f.write("Edge\n")
        f.write(f"{edge}\n\n")

        # Absorbing atom
        if absorber_idx is None:
            f.write('Z_absorber' + '\n')
            f.write(f'{z_absorber}' + 2*'\n')
        else:
            f.write('Absorber' + '\n') # 1-index
            f.write(f'{absorber_idx}' + 2*'\n')

        # Magnetic contributions
        if magnetism:
            f.write("Magnetism\n\n")

        if green:
            f.write("Green\n")
        if density_all:
            f.write("Density_all\n")
        if quadrupole:
            f.write("Quadrupole\n")
        if spherical:
            f.write("Spherical\n")
        if scf:
            f.write("SCF\n")
        f.write("\n")

        if all(ase_atoms.pbc):
            f.write("Crystal\n")
            f.write(" ".join(map(str, ase_atoms.cell.cellpar())) + "\n")
            positions = np.round(ase_atoms.get_scaled_positions(), 6)
        else:
            f.write("Molecule\n")
            cell_length = abs(ase_atoms.get_positions().max()) + abs(
                ase_atoms.get_positions().min()
            )
            f.write(f"{cell_length} {cell_length} {cell_length} 90 90 90\n")
            positions = np.round(ase_atoms.get_positions(), 6)

        for i, position in enumerate(positions):
            f.write(f"{atomic_numbers[i]} " + " ".join(map(str, position)) + "\n")

        f.write("\n")
        f.write("Convolution\n")
        f.write("End")


def get_normalized_xanes(
    conv_file: Path | str,
    pre_edge_width: float = 20.0,
    post_edge_width: float = 50.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize a XANES spectrum from an FDMNES convolution output file.

    Parameters
    ----------
    conv_file : Path or str
        Path to the FDMNES ``*_conv.txt`` output file.
    pre_edge_width : float
        Width of the pre-edge region in eV for baseline fitting.
    post_edge_width : float
        Width of the post-edge region in eV for step normalization.

    Returns
    -------
    normalized : np.ndarray
        (N, 2) array of [energy, normalized_mu].
    raw : np.ndarray
        (N, 2) array of [energy, raw_mu] as read from the file.
    """
    energy_xas = np.loadtxt(conv_file, skiprows=1)

    E = energy_xas[:, 0].astype(float)
    mu = energy_xas[:, 1].astype(float)

    if E.min() <= 0.0 <= E.max():
        E0 = 0.0
    else:
        dmu_dE = np.gradient(mu, E)
        E0 = E[np.argmax(dmu_dE)]

    pre_mask = E <= (E0 - pre_edge_width)
    post_mask = E >= (E0 + post_edge_width)

    if np.sum(pre_mask) < 2 or np.sum(post_mask) < 2:
        logger.warning(
            "Too few points in pre-edge or post-edge regions (pre-edge points: %d, post-edge points: %d). "
            "Normalizing by maximum mu value as fallback.",
            np.sum(pre_mask),
            np.sum(post_mask),
        )
        max_mu = np.max(mu)
        mu_norm = mu / max_mu if max_mu > 0 else mu
    else:
        try:
            m_pre, b_pre = np.polyfit(E[pre_mask], mu[pre_mask], 1)
            m_post, b_post = np.polyfit(E[post_mask], mu[post_mask], 1)

            pre_line = m_pre * E + b_pre
            mu_corr = mu - pre_line

            step = (m_post * E0 + b_post) - (m_pre * E0 + b_pre)
            if np.isclose(step, 0):
                raise ValueError("Step height is close to zero.")
            mu_norm = mu_corr / step
        except Exception as e:
            logger.warning(
                "XANES normalization fit failed: %s. Normalizing by maximum mu value as fallback.",
                e,
            )
            max_mu = np.max(mu)
            mu_norm = mu / max_mu if max_mu > 0 else mu

    return np.column_stack([E, mu_norm]), energy_xas


def extract_conv(fdmnes_output_dir: Path | str) -> dict:
    """Extract all convolution output files from an FDMNES run directory.

    Parameters
    ----------
    fdmnes_output_dir : Path or str
        Directory containing FDMNES output files.

    Returns
    -------
    dict
        Mapping of index to (N, 2) arrays of [energy, mu].
    """
    if not isinstance(fdmnes_output_dir, Path):
        fdmnes_output_dir = Path(fdmnes_output_dir)

    energy_xas = {}
    for i, conv_file in enumerate(fdmnes_output_dir.glob("*conv.txt")):
        energy_xas[i] = np.loadtxt(conv_file, skiprows=1)

    return energy_xas


def is_calc_done(run_dir: Path | str, min_size_bytes: int = 1024) -> bool:
    """Return True if a successful FDMNES convolution output already exists."""
    run_path = Path(run_dir)
    conv = next(run_path.glob("*_conv.txt"), None)
    return conv is not None and conv.stat().st_size > min_size_bytes


def _is_supported_xanes_structure_file(path: Path) -> bool:
    """Return True for structure files supported by the XANES batch loader."""
    name = path.name.lower()
    suffix = path.suffix.lower()
    return path.is_file() and (
        suffix in SUPPORTED_XANES_STRUCTURE_SUFFIXES or name in {"poscar", "contcar"}
    )


def _collect_xanes_batch_structures(
    input_source: str | list[str],
    ase_db_selection: str = "",
) -> tuple[list[dict], Path]:
    """Collect XANES batch structures from files, directories, or ASE databases."""
    structures: list[dict] = []

    if isinstance(input_source, list):
        structure_files = [Path(p).resolve() for p in input_source]
        missing = [str(p) for p in structure_files if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"The following input structures are missing: {missing}"
            )
        invalid = [
            str(p)
            for p in structure_files
            if not _is_supported_xanes_structure_file(p)
        ]
        if invalid:
            raise ValueError(
                "Only structure files are allowed in an explicit input list. "
                f"Unsupported entries: {invalid}"
            )
        output_root = structure_files[0].parent if structure_files else Path.cwd()
        for struct_path in structure_files:
            atoms = ase_read(str(struct_path))
            structures.append(
                {
                    "label": struct_path.stem,
                    "source": str(struct_path),
                    "atoms": atoms,
                }
            )
        return structures, output_root

    input_path = Path(input_source).resolve()

    if input_path.is_dir():
        structure_files = sorted(
            [
                p.resolve()
                for p in input_path.iterdir()
                if _is_supported_xanes_structure_file(p)
            ],
            key=lambda p: p.name.lower(),
        )
        if not structure_files:
            raise ValueError(f"No supported structure files found in '{input_path}'.")
        for struct_path in structure_files:
            atoms = ase_read(str(struct_path))
            structures.append(
                {
                    "label": struct_path.stem,
                    "source": str(struct_path),
                    "atoms": atoms,
                }
            )
        return structures, input_path

    if input_path.is_file() and input_path.suffix.lower() == ".db":
        from ase.db import connect as ase_db_connect

        db = ase_db_connect(str(input_path))
        selection = ase_db_selection or None
        rows = list(db.select(selection=selection)) if selection else list(db.select())
        if not rows:
            selector_text = (
                f" using selection '{ase_db_selection}'" if ase_db_selection else ""
            )
            raise ValueError(
                f"No structures found in ASE database '{input_path}'{selector_text}."
            )

        for row in rows:
            atoms = row.toatoms()
            atoms.info.setdefault("ase_db_id", row.id)
            atoms.info.setdefault("ase_db_source", str(input_path))
            formula = getattr(row, "formula", None) or atoms.get_chemical_formula()
            structures.append(
                {
                    "label": f"db_{row.id}_{formula}",
                    "source": f"{input_path}::id={row.id}",
                    "atoms": atoms,
                }
            )
        return structures, input_path.parent

    if input_path.is_file():
        if not _is_supported_xanes_structure_file(input_path):
            raise ValueError(
                f"Unsupported XANES input file '{input_path}'. "
                "Expected a structure file or an ASE database (.db)."
            )
        atoms = ase_read(str(input_path))
        structures.append(
            {
                "label": input_path.stem,
                "source": str(input_path),
                "atoms": atoms,
            }
        )
        return structures, input_path.parent

    raise ValueError(f"'{input_source}' is not a valid XANES input source.")


def _write_prepared_xanes_batch(
    structures: list[dict],
    root_dir: Path,
    z_absorber: Optional[int] = None,
    absorber_idx: Optional[int] = None,
    radius: float = 6.0,
    energy_range: Optional[List[float]] = None,
    magnetism: bool = False,
    skip_completed: bool = True,
    edge: str = "K",
    green: bool = True,
    density_all: bool = True,
    quadrupole: bool = True,
    spherical: bool = True,
    scf: bool = True,
) -> dict:
    """Prepare per-structure FDMNES input directories for a XANES batch."""
    root_dir = Path(root_dir).resolve()
    runs_dir = root_dir / "fdmnes_batch_runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    jobs = []
    n_prepared = 0
    n_skipped = 0

    for i, structure in enumerate(structures):
        atoms = structure["atoms"]
        run_dir = runs_dir / f"run_{i}"
        run_dir.mkdir(parents=True, exist_ok=True)

        current_z = (
            z_absorber
            if z_absorber is not None
            else int(max(atoms.get_atomic_numbers()))
        )

        job_meta = {
            "structure": structure["label"],
            "source": structure["source"],
            "run_dir": str(run_dir),
            "z_absorber": current_z,
            "absorber_idx": absorber_idx,
            "edge": edge,
            "green": green,
            "density_all": density_all,
            "quadrupole": quadrupole,
            "spherical": spherical,
            "scf": scf,
            "energy_range": energy_range or CHEMGRAPH_DEFAULT_ENERGY_RANGE,
        }

        if skip_completed and is_calc_done(run_dir):
            job_meta["status"] = "skipped_existing"
            jobs.append(job_meta)
            n_skipped += 1
            continue

        write_fdmnes_input(
            ase_atoms=atoms,
            z_absorber=current_z,
            absorber_idx=absorber_idx,
            input_file_dir=run_dir,
            radius=radius,
            energy_range=energy_range,
            magnetism=magnetism,
            edge=edge,
            green=green,
            density_all=density_all,
            quadrupole=quadrupole,
            spherical=spherical,
            scf=scf,
        )

        formula = atoms.get_chemical_formula()
        mp_id = atoms.info.get("MP-id", atoms.info.get("ase_db_id", "local"))
        pkl_filename = f"Z{current_z}_{mp_id}_{formula}.pkl"
        with open(run_dir / pkl_filename, "wb") as f:
            pickle.dump(atoms, f)

        with open(run_dir / "run_metadata.json", "w", encoding="utf-8") as f:
            json.dump(job_meta, f, indent=2)

        job_meta["status"] = "prepared"
        jobs.append(job_meta)
        n_prepared += 1

    return {
        "root_dir": str(root_dir),
        "runs_dir": str(runs_dir),
        "jobs": jobs,
        "n_total": len(jobs),
        "n_prepared": n_prepared,
        "n_skipped": n_skipped,
    }


def prepare_xanes_batch(
    input_source: str | list[str],
    z_absorber: Optional[int] = None,
    absorber_idx: Optional[int] = None,
    radius: float = 6.0,
    energy_range: Optional[List[float]] = None,
    magnetism: bool = False,
    output_dir: Optional[str] = None,
    ase_db_selection: str = "",
    skip_completed: bool = True,
    edge: str = "K",
    green: bool = True,
    density_all: bool = True,
    quadrupole: bool = True,
    spherical: bool = True,
    scf: bool = True,
) -> dict:
    """Prepare batch XANES/FDMNES run directories from a file, folder, or ASE DB."""
    structures, default_root = _collect_xanes_batch_structures(
        input_source=input_source,
        ase_db_selection=ase_db_selection,
    )
    root_dir = Path(output_dir).resolve() if output_dir is not None else default_root
    return _write_prepared_xanes_batch(
        structures=structures,
        root_dir=root_dir,
        z_absorber=z_absorber,
        absorber_idx=absorber_idx,
        radius=radius,
        energy_range=energy_range,
        magnetism=magnetism,
        skip_completed=skip_completed,
        edge=edge,
        green=green,
        density_all=density_all,
        quadrupole=quadrupole,
        spherical=spherical,
        scf=scf,
    )


# -----------------------------------------------------------------------------
# Core Workflow Functions
# -----------------------------------------------------------------------------


def run_xanes_core(params: xanes_input_schema) -> dict:
    """Run a single XANES/FDMNES calculation for one structure.

    This is the core function analogous to ``run_graspa_core``. It:
    1. Reads the input structure file via ASE.
    2. Creates FDMNES input files via ``write_fdmnes_input``.
    3. Runs FDMNES via subprocess.
    4. Parses the convolution output if available.

    Parameters
    ----------
    params : xanes_input_schema
        Input parameters for the FDMNES calculation.

    Returns
    -------
    dict
        Result dictionary with keys: status, output_dir, conv_data (if success),
        error (if failure).
    """
    fdmnes_exe = os.environ.get("FDMNES_EXE")
    if not fdmnes_exe:
        raise ValueError(
            "FDMNES_EXE environment variable is not set. "
            "Set it to the path of the FDMNES executable."
        )

    input_path = Path(params.input_structure_file).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input structure file not found: {input_path}")

    atoms = ase_read(str(input_path))

    # Determine output directory
    if params.output_dir is not None:
        run_dir = Path(params.output_dir).resolve()
    else:
        run_dir = input_path.parent / f"fdmnes_{input_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write FDMNES input files
    write_fdmnes_input(
        ase_atoms=atoms,
        z_absorber=params.z_absorber,
        absorber_idx=params.absorber_idx,
        input_file_dir=run_dir,
        radius=params.radius,
        energy_range=params.energy_range,
        magnetism=params.magnetism,
        edge=params.edge,
        green=params.green,
        density_all=params.density_all,
        quadrupole=params.quadrupole,
        spherical=params.spherical,
        scf=params.scf,
    )

    # Save the atoms object alongside the inputs for provenance
    formula = atoms.get_chemical_formula()
    z_abs = params.z_absorber or int(atoms.get_atomic_numbers().max())
    mp_id = atoms.info.get("MP-id", "local")
    pkl_filename = f"Z{z_abs}_{mp_id}_{formula}.pkl"
    with open(run_dir / pkl_filename, "wb") as f:
        pickle.dump(atoms, f)

    # Run FDMNES
    logger.info("Running FDMNES in %s", run_dir)
    with (
        open(run_dir / "fdmnes_stdout.txt", "w") as fp_out,
        open(run_dir / "fdmnes_stderr.txt", "w") as fp_err,
    ):
        proc = subprocess.run(
            fdmnes_exe,
            cwd=str(run_dir),
            stdout=fp_out,
            stderr=fp_err,
            shell=True,
        )

    if proc.returncode != 0:
        logger.error(
            "FDMNES failed with return code %d in %s", proc.returncode, run_dir
        )
        return {
            "status": "failure",
            "output_dir": str(run_dir),
            "error": f"FDMNES exited with return code {proc.returncode}",
        }

    # Parse results
    conv_data = extract_conv(run_dir)
    if not conv_data:
        logger.warning("No convolution output found in %s", run_dir)
        return {
            "status": "failure",
            "output_dir": str(run_dir),
            "error": "No *conv.txt output files found after FDMNES execution.",
        }

    logger.info("FDMNES completed successfully in %s", run_dir)
    return {
        "status": "success",
        "output_dir": str(run_dir),
        "n_conv_files": len(conv_data),
    }


def fetch_materials_project_data(
    params: mp_query_schema,
    db_path: Path,
) -> dict:
    """Fetch optimized structures from Materials Project.

    Parameters
    ----------
    params : mp_query_schema
        Query parameters including chemical formulas and API key.
    db_path : Path
        Directory to save the fetched structures.

    Returns
    -------
    dict
        atoms_list : list[Atoms]    — fetched ASE Atoms objects
        structure_files : list[str] — absolute paths to saved CIF files
        pickle_file : str           — absolute path to atoms_db.pkl
        n_structures : int          — number of structures fetched
    """
    from mp_api.client import MPRester
    from pymatgen.io.ase import AseAtomsAdaptor

    api_key = params.mp_api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        raise ValueError(
            "No Materials Project API key provided. "
            "Pass it via mp_api_key or set the MP_API_KEY environment variable."
        )

    logger.info("Fetching data from Materials Project for: %s", params.chemsys)
    atoms_list = []

    with MPRester(api_key) as mpr:
        doc_list = mpr.materials.summary.search(
            fields=["material_id", "structure"],
            energy_above_hull=(0, params.energy_above_hull),
            formula=params.chemsys,
            deprecated=False,
        )

        for doc in doc_list:
            ase_atoms = AseAtomsAdaptor.get_atoms(doc.structure)
            ase_atoms.info.update({"MP-id": str(doc.material_id)})
            atoms_list.append(ase_atoms)

    if not db_path.exists():
        db_path.mkdir(parents=True)

    # Save pickle database
    pkl_path = db_path / "atoms_db.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(atoms_list, f)

    # Save individual CIF files
    structure_files = []
    for atoms in atoms_list:
        mp_id = atoms.info.get("MP-id", "unknown")
        formula = atoms.get_chemical_formula()
        cif_path = db_path / f"{mp_id}_{formula}.cif"
        ase_write(str(cif_path), atoms)
        structure_files.append(str(cif_path))

    logger.info(
        "Saved %d structures (%s) and pickle database to %s",
        len(atoms_list),
        [Path(f).name for f in structure_files],
        db_path,
    )

    return {
        "atoms_list": atoms_list,
        "structure_files": structure_files,
        "pickle_file": str(pkl_path),
        "n_structures": len(atoms_list),
    }


def create_fdmnes_inputs(
    root_dir: Path,
    atoms_list: Optional[List[Atoms]] = None,
    z_absorber: Optional[int] = None,
    absorber_idx: Optional[int] = None,
    radius: float = 6.0,
    energy_range: Optional[List[float]] = None,
    magnetism: bool = False,
    edge: str = "K",
    green: bool = True,
    density_all: bool = True,
    quadrupole: bool = True,
    spherical: bool = True,
    scf: bool = True,
) -> Path:
    """Create FDMNES input files for a batch of structures.

    Parameters
    ----------
    root_dir : Path
        Root directory for the batch. A ``fdmnes_batch_runs`` subdirectory
        will be created containing per-structure run directories.
    atoms_list : list[ase.Atoms], optional
        Structures to process. If None, loads from ``root_dir/atoms_db.pkl``.
    z_absorber : int, optional
        Atomic number of the absorbing atom. Defaults to heaviest per structure.
    absorber_idx : int, optional
        1-based index of the absorbing atom in the structure.
        If provided, the Absorber keyword is used instead of Z_absorber.
    radius : float
        Cluster radius in Angstrom.
    energy_range : list of float, optional
        Values written under the FDMNES Range keyword.
    magnetism : bool
        Enable magnetic contributions.
    edge : str
        Absorption edge (e.g. K, L1, L2, L3, M1). Defaults to 'K'.
    green : bool
        Use Green function method for calculation. Default True.
    density_all : bool
        Compute full electron densities. Default True.
    quadrupole : bool
        Include quadrupole transition transitions. Default True.
    spherical : bool
        Assume spherical atoms. Default True.
    scf : bool
        Perform self-consistent field calculation. Default True.

    Returns
    -------
    Path
        Path to the ``fdmnes_batch_runs`` directory.
    """
    logger.info("Creating FDMNES inputs in %s", root_dir)
    root_dir = Path(root_dir).resolve()

    if atoms_list is None:
        db_path = root_dir / "atoms_db.pkl"
        if not db_path.exists():
            raise FileNotFoundError(f"No atoms provided and {db_path} not found.")
        with open(db_path, "rb") as f:
            atoms_list = pickle.load(f)

    structures = []
    for i, atoms in enumerate(atoms_list):
        label = atoms.info.get("MP-id", atoms.get_chemical_formula())
        structures.append(
            {
                "label": f"atoms_{i}_{label}",
                "source": f"atoms_list[{i}]",
                "atoms": atoms,
            }
        )

    batch = _write_prepared_xanes_batch(
        structures=structures,
        root_dir=root_dir,
        z_absorber=z_absorber,
        absorber_idx=absorber_idx,
        radius=radius,
        energy_range=energy_range,
        magnetism=magnetism,
        skip_completed=False,
        edge=edge,
        green=green,
        density_all=density_all,
        quadrupole=quadrupole,
        spherical=spherical,
        scf=scf,
    )
    return Path(batch["runs_dir"])


def expand_database_results(root_dir: Path, runs_dir: Path) -> None:
    """Expand the atoms database with XANES convolution results.

    For each completed run directory, loads the pickled Atoms object,
    attaches the FDMNES convolution data to ``atoms.info``, and saves
    all expanded structures to ``root_dir/atoms_db_expanded.pkl``.

    Parameters
    ----------
    root_dir : Path
        Root directory where the expanded database will be saved.
    runs_dir : Path
        Directory containing ``run_*`` subdirectories with FDMNES outputs.
    """
    logger.info("Expanding database with XANES results...")
    expanded_atoms_list = []

    for sub_dir in sorted(runs_dir.glob("run_*")):
        atoms_pkl_files = list(sub_dir.glob("*.pkl"))
        if not atoms_pkl_files:
            continue

        with open(atoms_pkl_files[0], "rb") as f:
            ase_atoms = pickle.load(f)

        conv_data = extract_conv(fdmnes_output_dir=sub_dir)
        ase_atoms.info.update({"FDMNES-xanes": conv_data})
        expanded_atoms_list.append(ase_atoms)

    with open(root_dir / "atoms_db_expanded.pkl", "wb") as f:
        pickle.dump(expanded_atoms_list, f)

    logger.info(
        "Saved %d expanded structures to %s",
        len(expanded_atoms_list),
        root_dir / "atoms_db_expanded.pkl",
    )


def plot_xanes_results(root_dir: Path, runs_dir: Path) -> dict:
    """Generate normalized XANES plots for completed FDMNES calculations.

    For each run directory containing a ``*_conv.txt`` file, produces
    a ``xanes_plot.png`` with the normalized absorption spectrum.

    Parameters
    ----------
    root_dir : Path
        Root data directory (unused currently, reserved for summary plots).
    runs_dir : Path
        Directory containing ``run_*`` subdirectories with FDMNES outputs.

    Returns
    -------
    dict
        plot_files : list[str]  — absolute paths to generated plot images
        n_plots : int           — number of plots successfully generated
        n_failed : int          — number of runs that failed to plot
        failed : list[str]      — names of run directories that failed
    """
    import matplotlib.pyplot as plt

    logger.info("Plotting XANES results from %s", runs_dir)

    plot_files = []
    failed = []

    for sub_dir in sorted(runs_dir.glob("run_*")):
        conv_file = next(sub_dir.glob("*_conv.txt"), None)
        if conv_file:
            try:
                norm_energy, _raw = get_normalized_xanes(conv_file)
                plot_path = sub_dir / "xanes_plot.png"
                plt.figure()
                plt.plot(norm_energy[:, 0], norm_energy[:, 1], label=sub_dir.name)
                plt.xlabel("Energy [eV]")
                plt.ylabel("Normalized Absorption")
                plt.title(f"XANES for {sub_dir.name}")
                plt.legend()
                plt.savefig(plot_path, dpi=150)
                plt.close()
                plot_files.append(str(plot_path))
                logger.info("Plotted %s", sub_dir.name)
            except Exception as e:
                logger.error("Failed to plot %s: %s", sub_dir.name, e)
                failed.append(sub_dir.name)

    return {
        "plot_files": plot_files,
        "n_plots": len(plot_files),
        "n_failed": len(failed),
        "failed": failed,
    }


# -----------------------------------------------------------------------------
# Data directory helper
# -----------------------------------------------------------------------------


def _get_data_dir() -> Path:
    """Return the working data directory for XANES workflows."""
    cwd = Path.cwd()
    if "PBS_O_WORKDIR" in os.environ:
        cwd = Path(os.environ["PBS_O_WORKDIR"])

    data_dir = cwd / "xanes_data"
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
    return data_dir


@tool
def run_xanes(params: xanes_input_schema) -> str:
    """Run a single XANES/FDMNES calculation for one structure file.

    This tool reads the structure, generates FDMNES input files, runs FDMNES,
    and returns the result status. Requires the FDMNES_EXE environment variable.
    """
    result = run_xanes_core(params)
    if result["status"] == "success":
        return (
            f"XANES calculation completed successfully. "
            f"Output directory: {result['output_dir']}. "
            f"Found {result['n_conv_files']} convolution output(s)."
        )
    else:
        raise RuntimeError(
            f"FDMNES calculation failed in {result['output_dir']}: "
            f"{result.get('error', 'unknown error')}"
        )


@tool
def fetch_xanes_data(params: mp_query_schema) -> str:
    """Fetch optimized bulk structures from Materials Project for XANES analysis.

    Requires a Materials Project API key via the mp_api_key parameter
    or the MP_API_KEY environment variable.
    """
    data_dir = _get_data_dir()
    result = fetch_materials_project_data(params, data_dir)
    return (
        f"Fetched {result['n_structures']} structures for {params.chemsys} "
        f"into {data_dir}. "
        f"Structure files: {result['structure_files']}"
    )


@tool
def plot_xanes_data(runs_dir: str) -> str:
    """Generate normalized XANES plots for completed FDMNES calculations.

    Produces a xanes_plot.png in each run directory that contains
    FDMNES convolution output files (*_conv.txt).

    Parameters
    ----------
    runs_dir : str
        Path to the directory containing ``run_*`` subdirectories
        with FDMNES outputs.
    """
    runs_path = Path(runs_dir)
    if not runs_path.is_dir():
        raise ValueError(f"'{runs_dir}' is not a valid directory.")

    data_dir = _get_data_dir()
    result = plot_xanes_results(data_dir, runs_path)
    if result["n_failed"] > 0:
        return (
            f"Generated {result['n_plots']} plot(s), "
            f"{result['n_failed']} failed ({result['failed']}). "
            f"Plot files: {result['plot_files']}"
        )
    return f"Generated {result['n_plots']} plot(s). Plot files: {result['plot_files']}"
