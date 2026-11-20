"""All plotting code, one module per section of the report. Each function returns a matplotlib figure."""

from rydberg.config import FIGURES_DIR

WAIT_COLORS = ["tab:blue", "tab:orange", "tab:red"]
METHOD_COLORS = {"A. isolated": "tab:gray", "B. per degree": "tab:green", "C. GNN": "tab:purple"}


def save_figure(figure, file_name):
  """Saves the figure in reports/figures/ (used by the markdown report)."""
  FIGURES_DIR.mkdir(parents=True, exist_ok=True)
  figure.savefig(FIGURES_DIR / file_name, dpi=130, bbox_inches="tight")
  return figure
