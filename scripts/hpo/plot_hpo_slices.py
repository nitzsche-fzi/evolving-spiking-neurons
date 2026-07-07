"""Renders an Optuna slice plot for an HPO study to an HTML file.

Loads the single study stored in a SQLite Optuna database and writes its slice
plot to ``results/optuna/slice_plot.html``.

Example:
    python3 scripts/hpo/plot_hpo_slices.py path/to/study.db
"""

# Ensure the repo root is importable so `from lib...` / `from analysis...`
# work regardless of the directory this script is invoked from.
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)))

import optuna
import optuna.visualization as vis
import sys
from pathlib import Path

def main():
    if len(sys.argv) != 2:
        print("Usage: python3 scripts/hpo/plot_hpo_slices.py path/to/study.db")
        sys.exit(1)

    db_path = sys.argv[1]
    storage = f"sqlite:///{db_path}"

    # Get the only study in the database
    summaries = optuna.study.get_all_study_summaries(storage=storage)
    if len(summaries) == 0:
        print("No studies found in the database.")
        sys.exit(1)
    if len(summaries) > 1:
        print("Multiple studies found. Specify the study name manually.")
        sys.exit(1)

    study_name = summaries[0].study_name

    # Load and plot
    study = optuna.load_study(study_name=study_name, storage=storage)
    fig = vis.plot_slice(study)

    out_dir = Path("results/optuna")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html("results/optuna/slice_plot.html")

if __name__ == "__main__":
    main()
