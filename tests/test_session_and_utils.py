import parastat as ms
import pandas as pd
import numpy as np
import os


def test_session_and_io(tmp_path):
    s = ms.Session(device="cpu")
    # 准备一个 CSV
    p = tmp_path / "d.csv"
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df.to_csv(p, index=False)
    d2 = s.read_csv(str(p))
    assert list(d2.columns) == ["a", "b"]
    info = s.info()
    assert "backend" in info
    assert "session_device" in info
    s.close()


def test_utils():
    ms.set_seed(123)
    v = ms.version()
    assert isinstance(v, str)
    di = ms.device_info()
    assert isinstance(di, dict)



