import os
import pytest
from src.quant_trader.utils.config import load_config

@pytest.mark.smoke
def test_load_config():
    cfg_path = "configs/base.yaml"
    if not os.path.exists(cfg_path):
        pytest.skip(f"Config file not found: {cfg_path}")

    cfg = load_config(cfg_path)
    assert "project" in cfg
    assert "data" in cfg
    assert len(cfg["data"].get("tickers", [])) > 0

