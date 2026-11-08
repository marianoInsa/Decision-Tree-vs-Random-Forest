"""Genera observaciones del informe a partir de las tablas exportadas actuales."""
from pathlib import Path
import json


def actualizar_observaciones():
    """Escribe cifras calculadas con los mismos resúmenes que alimentan los gráficos."""
    carpeta = Path(__file__).resolve().parent
    resultados = json.loads((carpeta / "resultados.json").read_text(encoding="utf-8"))
    texto = []
    observaciones_finales = []
    for nombre, etiqueta in (("lenses", "Lenses"), ("iris", "Iris"), ("heart", "Cleveland")):
        filas = {fila["modelo"]: fila for fila in resultados["finales"]
                 if fila["dataset"] == nombre and fila["conjunto"] == "prueba"}
        observaciones_finales.append(f"En {etiqueta}, el F1 macro final es "
            f"{filas['C4.5']['f1_macro']:.3f} para C4.5 y {filas['Random Forest']['f1_macro']:.3f} para Random Forest.")
    observaciones_finales.append("Estas cifras corresponden a la selección interna por validación; no al máximo de los barridos de prueba.")
    texto.append(" ".join(observaciones_finales))
    observaciones_nulos = []
    for nombre in ("lenses", "iris", "heart"):
        etiquetas = {"lenses": "Lenses", "iris": "Iris", "heart": "Cleveland"}
        partes = []
        for modelo in ("C4.5", "Random Forest"):
            filas = [fila for fila in resultados["faltantes"] if fila["dataset"] == nombre
                     and fila["modelo"] == modelo and fila["conjunto"] == "prueba"]
            inicial = next(fila for fila in filas if fila["fraccion_nulos_objetivo"] == 0)
            final = next(fila for fila in filas if fila["fraccion_nulos_objetivo"] == 0.3)
            partes.append(f"{modelo} pasa de {inicial['f1_macro_media']:.3f} a {final['f1_macro_media']:.3f}")
        observaciones_nulos.append(f"En {etiquetas[nombre]}, al pasar de cero a treinta por ciento de ausencias adicionales, "
                     + "; ".join(partes) + " de F1 macro medio.")
    observaciones_nulos.append("La comparación conserva parámetros y casos; no se supone que el efecto sea monótono ni que se repita en otros problemas.")
    texto.append(" ".join(observaciones_nulos))
    filas = [fila for fila in resultados["barridos"] if fila["dataset"] == "heart"
             and fila["modelo"] == "C4.5" and fila["parametro_variado"] == "filtro_ganancia"
             and fila["conjunto"] == "prueba"]
    sin_filtro = next(fila for fila in filas if fila["valor_parametro"] == "false")
    con_filtro = next(fila for fila in filas if fila["valor_parametro"] == "true")
    texto.append(f"En el barrido descriptivo de Cleveland, el árbol sin filtro obtiene "
        f"{sin_filtro['accuracy_media']:.3f} de tasa de acierto media, con desvío {sin_filtro['accuracy_desvio']:.3f}; "
        f"con filtro obtiene {con_filtro['accuracy_media']:.3f}, con desvío {con_filtro['accuracy_desvio']:.3f}. "
        "Se informa la variante sin activarla por defecto. Ambas condiciones conservan los demás parámetros y las particiones. "
        "La comparación describe esta muestra y no selecciona el uso del filtro mediante los resultados de prueba.")
    (carpeta / "observaciones_resultados.tex").write_text("\n\n".join(texto)+"\n",encoding="utf-8")


if __name__ == "__main__":
    actualizar_observaciones()
