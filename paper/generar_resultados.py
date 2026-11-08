"""Regenera los experimentos del notebook y sus tablas y figuras del informe.

Se ejecuta desde la raíz con Python. Se usa el código real del notebook 03,
sin mantener otro algoritmo ni otra batería. Esta ejecución puede tardar varios
minutos; no utiliza los resultados históricos de los PR anteriores.
"""
from pathlib import Path
import json
import os

import matplotlib
matplotlib.use("Agg")
from IPython.utils.capture import capture_output


def generar_resultados():
    """Ejecuta los experimentos actuales y conserva sus exportaciones reproducibles."""
    raiz = Path(__file__).resolve().parents[1]
    notebook = raiz / "notebooks/03_experimentos.ipynb"
    espacio = {"__name__": "experimentos_para_informe"}
    directorio_original = Path.cwd()
    try:
        os.chdir(raiz)
        for numero, celda in enumerate(json.loads(notebook.read_text(encoding="utf-8"))["cells"]):
            if celda["cell_type"] == "code":
                # Ocultar tablas y figuras en consola no modifica las métricas ni sus archivos.
                with capture_output():
                    exec(compile("".join(celda["source"]), f"{notebook}:celda_{numero}", "exec"), espacio)
                print(f"Celda {numero}: completada.", flush=True)
    finally:
        os.chdir(directorio_original)
    from paper.actualizar_informe import actualizar_observaciones
    actualizar_observaciones()
    return raiz / "paper/resultados.json"


if __name__ == "__main__":
    print(generar_resultados())
