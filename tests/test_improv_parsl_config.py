import sys
from types import ModuleType

import pytest

from chemgraph.hpc_configs.improv_parsl import get_improv_config


class _CapturedConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _install_fake_parsl(monkeypatch):
    parsl = ModuleType("parsl")
    parsl.__path__ = []

    config = ModuleType("parsl.config")
    config.Config = _CapturedConfig
    executors = ModuleType("parsl.executors")
    executors.HighThroughputExecutor = _CapturedConfig
    launchers = ModuleType("parsl.launchers")
    launchers.SingleNodeLauncher = _CapturedConfig
    providers = ModuleType("parsl.providers")
    providers.PBSProProvider = _CapturedConfig

    monkeypatch.setitem(sys.modules, "parsl", parsl)
    monkeypatch.setitem(sys.modules, "parsl.config", config)
    monkeypatch.setitem(sys.modules, "parsl.executors", executors)
    monkeypatch.setitem(sys.modules, "parsl.launchers", launchers)
    monkeypatch.setitem(sys.modules, "parsl.providers", providers)


def _clear_optional_improv_env(monkeypatch):
    for name in (
        "CHEMGRAPH_PBS_WALLTIME",
        "CHEMGRAPH_CPUS_PER_NODE",
        "CHEMGRAPH_MAX_BLOCKS",
        "CHEMGRAPH_INIT_BLOCKS",
        "CHEMGRAPH_MIN_BLOCKS",
        "CHEMGRAPH_DRAIN_PERIOD",
        "CHEMGRAPH_RETRIES",
        "CHEMGRAPH_WORKER_DEBUG",
        "PBS_NP",
    ):
        monkeypatch.delenv(name, raising=False)


def test_improv_config_uses_resilient_xanes_defaults(monkeypatch, tmp_path):
    _install_fake_parsl(monkeypatch)
    _clear_optional_improv_env(monkeypatch)
    monkeypatch.setenv("CHEMGRAPH_PBS_ACCOUNT", "test-account")

    config = get_improv_config(run_dir=str(tmp_path))
    executor = config.kwargs["executors"][0]
    provider = executor.kwargs["provider"]

    assert config.kwargs["retries"] == 1
    assert executor.kwargs["drain_period"] == 600
    assert provider.kwargs["walltime"] == "12:00:00"


def test_improv_config_allows_resilience_overrides(monkeypatch, tmp_path):
    _install_fake_parsl(monkeypatch)
    _clear_optional_improv_env(monkeypatch)
    monkeypatch.setenv("CHEMGRAPH_PBS_ACCOUNT", "test-account")
    monkeypatch.setenv("CHEMGRAPH_PBS_WALLTIME", "16:00:00")
    monkeypatch.setenv("CHEMGRAPH_DRAIN_PERIOD", "900")
    monkeypatch.setenv("CHEMGRAPH_RETRIES", "2")

    config = get_improv_config(run_dir=str(tmp_path))
    executor = config.kwargs["executors"][0]
    provider = executor.kwargs["provider"]

    assert config.kwargs["retries"] == 2
    assert executor.kwargs["drain_period"] == 900
    assert provider.kwargs["walltime"] == "16:00:00"


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CHEMGRAPH_DRAIN_PERIOD", "0", "must be greater than zero"),
        ("CHEMGRAPH_RETRIES", "-1", "must be zero or greater"),
    ],
)
def test_improv_config_rejects_invalid_resilience_values(
    monkeypatch,
    tmp_path,
    name,
    value,
    message,
):
    _install_fake_parsl(monkeypatch)
    _clear_optional_improv_env(monkeypatch)
    monkeypatch.setenv("CHEMGRAPH_PBS_ACCOUNT", "test-account")
    monkeypatch.setenv(name, value)

    with pytest.raises(ValueError, match=message):
        get_improv_config(run_dir=str(tmp_path))
