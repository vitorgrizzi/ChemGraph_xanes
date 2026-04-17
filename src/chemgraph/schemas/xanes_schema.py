import os
from typing import Optional

from pydantic import BaseModel, Field, model_validator


class xanes_input_schema(BaseModel):
    """Input schema for a single XANES/FDMNES calculation."""

    input_structure_file: str = Field(
        description="Path to the input structure file (CIF, POSCAR, XYZ, etc.)."
    )
    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory to write FDMNES input files and results. "
            "Defaults to a subdirectory next to the input structure."
        ),
    )
    z_absorber: Optional[int] = Field(
        default=None,
        description=(
            "Atomic number of the X-ray absorbing atom. "
            "Defaults to the heaviest element in the structure."
        ),
    )
    radius: float = Field(
        default=6.0,
        description="Cluster radius in Angstrom for the FDMNES calculation.",
    )
    magnetism: bool = Field(
        default=False,
        description="Enable magnetic contributions in the FDMNES calculation.",
    )


class xanes_input_schema_ensemble(BaseModel):
    """Input schema for ensemble XANES/FDMNES calculations via Parsl."""

    input_source: Optional[str] = Field(
        default=None,
        description=(
            "Path to a directory of structure files, a single structure file, "
            "or an ASE database (.db)."
        ),
    )
    input_structure_files: list[str] = Field(
        default_factory=list,
        description=(
            "Explicit list of individual structure file paths to run as an ensemble."
        ),
    )
    output_dir: Optional[str] = Field(
        default=None,
        description=(
            "Directory to write batch run directories and summary logs. "
            "Defaults to the input directory, structure parent directory, "
            "or ASE database parent directory."
        ),
    )
    z_absorber: Optional[int] = Field(
        default=None,
        description=(
            "Atomic number of the X-ray absorbing atom. "
            "Defaults to the heaviest element in each structure."
        ),
    )
    radius: float = Field(
        default=6.0,
        description="Cluster radius in Angstrom for the FDMNES calculation.",
    )
    magnetism: bool = Field(
        default=False,
        description="Enable magnetic contributions in the FDMNES calculation.",
    )
    ase_db_selection: str = Field(
        default="",
        description=(
            "Optional ASE database selection string used when "
            "input_source points to an ASE database."
        ),
    )
    skip_completed: bool = Field(
        default=True,
        description=(
            "Skip run directories that already contain a non-empty "
            "FDMNES convolution output."
        ),
    )
    fdmnes_exe: str = Field(
        default_factory=lambda: os.environ.get("FDMNES_EXE", "fdmnes"),
        description=(
            "Path to the FDMNES executable. "
            "Defaults to the FDMNES_EXE environment variable, or 'fdmnes'."
        ),
    )

    @model_validator(mode="after")
    def validate_input_choice(self):
        """Ensure exactly one ensemble input mode is selected."""
        has_source = bool(self.input_source)
        has_file_list = bool(self.input_structure_files)

        if has_source == has_file_list:
            raise ValueError(
                "Provide exactly one of `input_source` or `input_structure_files`."
            )

        return self

    def resolve_input_source(self) -> str | list[str]:
        """Return the effective input source expected by the batch helper."""
        if self.input_source:
            return self.input_source
        return self.input_structure_files


class mp_query_schema(BaseModel):
    """Input schema for fetching structures from Materials Project."""

    chemsys: list[str] = Field(
        description="Chemical formulas to search (e.g. ['Fe2O3', 'CoO']).",
    )
    mp_api_key: Optional[str] = Field(
        default=None,
        description=(
            "Materials Project API key. "
            "If not provided, falls back to the MP_API_KEY environment variable."
        ),
    )
    energy_above_hull: float = Field(
        default=0.001,
        description=(
            "Maximum energy above hull in eV/atom for filtering stable structures."
        ),
    )
