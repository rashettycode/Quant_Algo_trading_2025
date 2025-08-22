import sys
from pathlib import Path

# Ensure repo root is on sys.path when tests are run from anywhere
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.append(str(REPO))

from src.quant_trader.utils.config import load_config  # noqa: E402


def test_base_config_loads():
    cfg = load_config(str(REPO / "configs" / "base.yaml"))
    assert isinstance(cfg, dict)
    # basic keys we depend on
    assert "project" in cfg
    assert "data" in cfg
    assert "sources" in cfg
    assert "modes" in cfg
