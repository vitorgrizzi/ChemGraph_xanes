from chemgraph.tools.xanes_tools import _resolve_fdmnes_executable


def test_resolve_fdmnes_executable_uses_server_configuration(monkeypatch):
    configured_exe = "/opt/fdmnes/mpirun_fdmnes"
    monkeypatch.setenv("FDMNES_EXE", configured_exe)

    assert _resolve_fdmnes_executable("fdmnes") == configured_exe


def test_resolve_fdmnes_executable_preserves_explicit_path(monkeypatch):
    monkeypatch.setenv("FDMNES_EXE", "/opt/fdmnes/server_default")
    requested_exe = "/project/fdmnes/custom_launcher"

    assert _resolve_fdmnes_executable(requested_exe) == requested_exe


def test_resolve_fdmnes_executable_keeps_fallback_without_configuration(
    monkeypatch,
):
    monkeypatch.delenv("FDMNES_EXE", raising=False)

    assert _resolve_fdmnes_executable("fdmnes") == "fdmnes"
