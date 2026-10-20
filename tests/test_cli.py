from pathlib import Path

from lacc.cli import run_demo

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "demo.yaml"


def test_demo_writes_outputs(tmp_path):
    result = run_demo(CONFIG, tmp_path)
    assert (tmp_path / "demo_kpis.json").exists()
    assert (tmp_path / "demo_trace.png").exists()
    assert result["kpis"]["collision"] is False
