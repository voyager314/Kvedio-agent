import pytest


@pytest.fixture
def home(tmp_path, monkeypatch):
    """An isolated configuration home, so no test reads or writes the user's own directory."""
    target = tmp_path / "kvedio_home"
    monkeypatch.setenv("KVEDIO_AGENT_HOME", str(target))
    return target
