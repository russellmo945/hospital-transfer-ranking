"""Run the notebooks' Python cells headlessly; no Jupyter server is required."""

import json
import os
from pathlib import Path

# Bound parallelism and use file-based plotting for noninteractive runs.
for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ[variable] = "1"
os.environ["MPLBACKEND"] = "Agg"


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)
    import matplotlib.pyplot as plt

    def display(value):
        table = getattr(value, "data", value)
        print(table.to_string() if hasattr(table, "to_string") else str(table))

    for name in ["Preliminary_Data_Analysis", "Predictive_Model"]:
        notebook = json.loads((root / "notebooks" / f"{name}.ipynb").read_text())
        namespace = {"__name__": "__main__", "display": display}
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                exec(compile(source, f"{name}.ipynb:cell-{i}", "exec"), namespace)
        plt.close("all")
        print(f"Completed {name}", flush=True)
    print("Saved metrics, capture curve, and hospital ranking to runs/modeling/", flush=True)


if __name__ == "__main__":
    main()
