"""
In-process notebook runner for sandboxed execution without ZMQ socket overhead.
Executes each code cell sequentially in a shared global namespace, captures
stdout, stderr, text/plain, and matplotlib figures, and saves the executed
notebook with outputs populated.
"""

import ast
import io
import os
import sys
import time
import traceback
from pathlib import Path
import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ["MPLCONFIGDIR"] = str(PROJECT_ROOT / ".mplconfig")
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_notebook(nb_path: Path):
    nb_path = nb_path.resolve()
    print(f"\n=======================================================")
    print(f"Running notebook: {nb_path.name}")
    print(f"=======================================================")
    t0 = time.time()
    
    orig_cwd = os.getcwd()
    os.chdir(nb_path.parent)
    try:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = nbf.read(f, as_version=4)

        # Clean execution namespace
        glob = {
            "__name__": "__main__",
            "__file__": str(nb_path),
            "PROJECT_ROOT": PROJECT_ROOT,
        }
        
        execution_count = 1
        code_cells_count = sum(1 for c in nb.cells if c.cell_type == "code")
        current_code_idx = 0

        for i, cell in enumerate(nb.cells):
            if cell.cell_type != "code":
                continue
            
            current_code_idx += 1
            src = cell.source.strip()
            if not src:
                continue

            print(f"[{current_code_idx}/{code_cells_count}] Running code cell {i} ({len(src.splitlines())} lines)...")
            
            # Capture stdout and stderr
            stdout_trap = io.StringIO()
            stderr_trap = io.StringIO()
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # Track matplotlib figures
            plt.close("all")
            
            cell.outputs = []
            cell.execution_count = execution_count
            
            try:
                sys.stdout = stdout_trap
                sys.stderr = stderr_trap
                
                # Split last expression if it's an expression (to capture execute_result)
                parsed = ast.parse(src)
                last_expr = None
                if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                    last_expr = ast.Expression(parsed.body[-1].value)
                    parsed.body.pop()
                
                # Execute statement body
                if parsed.body:
                    code_obj = compile(parsed, filename=str(nb_path), mode="exec")
                    exec(code_obj, glob)
                
                # Eval last expression if present
                if last_expr is not None:
                    expr_code = compile(last_expr, filename=str(nb_path), mode="eval")
                    res = eval(expr_code, glob)
                    if res is not None:
                        cell.outputs.append(
                            nbf.v4.new_output(
                                output_type="execute_result",
                                data={"text/plain": repr(res)},
                                execution_count=execution_count,
                            )
                        )
                
            except Exception as e:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                print(f"ERROR in cell {i}: {e}")
                traceback.print_exc()
                # Record error in cell output
                tb = traceback.format_exc().splitlines()
                cell.outputs.append(
                    nbf.v4.new_output(
                        output_type="error",
                        ename=type(e).__name__,
                        evalue=str(e),
                        traceback=tb,
                    )
                )
                # Save partial outputs
                with open(nb_path, "w", encoding="utf-8") as f:
                    nbf.write(nb, f)
                raise e
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr

            # Check stdout / stderr
            out_text = stdout_trap.getvalue()
            err_text = stderr_trap.getvalue()
            if out_text:
                cell.outputs.insert(
                    0,
                    nbf.v4.new_output(output_type="stream", name="stdout", text=out_text)
                )
                print(out_text.rstrip())
            if err_text:
                cell.outputs.append(
                    nbf.v4.new_output(output_type="stream", name="stderr", text=err_text)
                )

            # Check figures created
            fig_nums = plt.get_fignums()
            for fnum in fig_nums:
                fig = plt.figure(fnum)
                buf = io.BytesIO()
                fig.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                import base64
                img_b64 = base64.b64encode(buf.read()).decode("ascii")
                cell.outputs.append(
                    nbf.v4.new_output(
                        output_type="display_data",
                        data={"image/png": img_b64, "text/plain": f"<Figure size {fig.get_size_inches()} with {len(fig.axes)} Axes>"},
                    )
                )
            plt.close("all")

            execution_count += 1

        with open(nb_path, "w", encoding="utf-8") as f:
            nbf.write(nb, f)

        elapsed = round(time.time() - t0, 1)
        print(f"SUCCESS: {nb_path.name} finished in {elapsed}s with all outputs saved!\n")
    finally:
        os.chdir(orig_cwd)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <path_to_notebook>")
        sys.exit(1)
    run_notebook(Path(sys.argv[1]))
