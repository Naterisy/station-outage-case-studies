"""Check the regenerated tables against frozen development results."""
from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
for name in ("kriging_dev", "historical_dev", "analog_dev", "cold_dev"):
    actual = pd.read_csv(root/name/"summary.csv")
    expected = pd.read_csv(root/"expected"/f"{name}.csv")
    pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=1e-9,atol=1e-10)
    print(f"{name}: matches frozen development table")
