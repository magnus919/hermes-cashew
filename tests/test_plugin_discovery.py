# tests/test_plugin_discovery.py
# PKG-02 verification: pyproject.toml declares [project.entry-points."hermes_agent.plugins"].
# REQUIRES: `pip install -e ".[dev]"` has been run (entry points are metadata-backed).
import importlib.metadata as im


def test_entry_point_registered():
    """PKG-02: pyproject.toml declares [project.entry-points."hermes_agent.plugins"]."""
    eps = im.entry_points(group="hermes_agent.plugins")
    names = {ep.name for ep in eps}
    assert "cashew" in names, f"cashew entry point missing; got {names}"


def test_entry_point_loads():
    """The entry-point target must be a loadable module with a callable register function."""
    eps = im.entry_points(group="hermes_agent.plugins")
    cashew_eps = [ep for ep in eps if ep.name == "cashew"]
    assert len(cashew_eps) == 1, f"expected 1 cashew entry point, got {len(cashew_eps)}"
    mod = cashew_eps[0].load()
    assert hasattr(mod, "register"), "entry point module must expose a register function"
    assert callable(mod.register), f"register must be callable, got {type(mod.register)}"


def test_module_importable():
    """PKG-01: plugins.memory.cashew is importable from the installed package."""
    import plugins.memory.cashew
    assert hasattr(plugins.memory.cashew, "CashewMemoryProvider")
    assert hasattr(plugins.memory.cashew, "register")
