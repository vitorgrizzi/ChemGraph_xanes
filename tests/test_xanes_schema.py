import pytest

from chemgraph.schemas.xanes_schema import xanes_input_schema_ensemble


def test_xanes_ensemble_schema_accepts_input_source():
    params = xanes_input_schema_ensemble(input_source="/tmp/structures.db")
    assert params.resolve_input_source() == "/tmp/structures.db"


def test_xanes_ensemble_schema_accepts_file_list():
    params = xanes_input_schema_ensemble(
        input_structure_files=["/tmp/a.cif", "/tmp/b.cif"]
    )
    assert params.resolve_input_source() == ["/tmp/a.cif", "/tmp/b.cif"]


def test_xanes_ensemble_schema_requires_exactly_one_mode():
    with pytest.raises(ValueError):
        xanes_input_schema_ensemble()

    with pytest.raises(ValueError):
        xanes_input_schema_ensemble(
            input_source="/tmp/structures.db",
            input_structure_files=["/tmp/a.cif"],
        )
