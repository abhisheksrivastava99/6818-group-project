"""Streamlit entrypoint."""

from pathlib import Path
import sys


# Streamlit deployments typically run `streamlit run app.py` from the repo root,
# so add the src layout explicitly when the package is not installed.
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from banksimfm.app.dashboard import run_dashboard


if __name__ == "__main__":
    run_dashboard()
