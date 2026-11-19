"""Tiny helper used by scripts/gen_*.py to write Jupyter notebooks."""

from pathlib import Path
import nbformat as nbf


def write_notebook(path: Path, cells):
    nb = nbf.v4.new_notebook()
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        }
    }
    nb["cells"] = []
    for kind, src in cells:
        src = src.strip("\n") + "\n"
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(src))
        else:
            nb.cells.append(nbf.v4.new_code_cell(src))
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    print("wrote", path, "cells=", len(nb.cells))
