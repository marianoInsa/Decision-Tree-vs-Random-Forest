"""Genera los resultados y las figuras de la sección de simulaciones del informe.

Para qué: el informe no debe tener números escritos a mano. Este script corre
todas las pruebas y guarda lo que el texto cita, así cualquiera del equipo puede
repetirlo y obtener lo mismo.

Cómo se usa, desde la raíz del repositorio:

    python paper/generar_resultados.py

Salida: paper/resultados.json y las figuras de paper/figuras/.
"""
import json
from pathlib import Path

import matplotlib

# "Agg" dibuja sin abrir ventanas: el script no necesita pantalla.
matplotlib.use("Agg")

RAIZ = Path(__file__).resolve().parents[1]
FIGURAS = RAIZ / "paper/figuras"

# Se reutilizan las funciones de la interfaz para que el informe y la interfaz
# calculen exactamente lo mismo, sin mantener dos versiones del experimento.
interfaz = {}
notebook = RAIZ / "notebooks/04_interfaz.ipynb"
for celda in json.loads(notebook.read_text(encoding="utf-8"))["cells"]:
    if "interfaz-definiciones" in celda.get("metadata", {}).get("tags", []):
        exec(compile("".join(celda["source"]), str(notebook), "exec"), interfaz)

DATASET = "heart"
# Diez particiones distintas: con 61 casos de prueba, una sola partición puede
# dar un resultado engañoso por azar. La media y el desvío lo muestran mejor.
REPETICIONES = 10
SEMILLA = 42

# Valores por defecto: árbol sin límites y bosque de 100 árboles.
C45_BASE = {"max_depth": None, "min_samples_split": 2, "threshold": 0.0, "min_gain_ratio": 0.0}
RF_BASE = {"n_estimators": 100, "max_features": "sqrt", "criterion": "gini",
           "bootstrap": True, "max_depth": None}

# Parámetros numéricos: se dibuja una curva con cada uno.
BARRIDOS = {
    "C4.5": {
        "max_depth": [1, 2, 3, 4, 5, 6, 8, 10],
        "min_samples_split": [2, 5, 10, 20, 40],
        "threshold": [0, 0.01, 0.02, 0.05, 0.1],
        "min_gain_ratio": [0, 0.02, 0.05, 0.1, 0.2],
    },
    "Random Forest": {
        "n_estimators": [10, 25, 50, 100, 200],
        "max_depth": [1, 2, 3, 4, 5, 6, 8, 10],
    },
}

# Nombre de cada parámetro en los ejes de las figuras del informe.
ETIQUETAS = {
    "max_depth": "profundidad máxima", "min_samples_split": "masa mínima para dividir",
    "threshold": "ganancia mínima (bits)", "min_gain_ratio": "tasa de ganancia mínima",
    "n_estimators": "cantidad de árboles",
}

# Parámetros que no son números: se compara cada opción por separado.
OPCIONES_RF = {
    "max_features": ["sqrt", "log2", None],
    "criterion": ["gini", "entropy"],
    "bootstrap": [True, False],
}


def resumen_base():
    """Compara ambos modelos con sus valores por defecto."""
    detalle = interfaz["comparar"](DATASET, C45_BASE, RF_BASE, REPETICIONES, SEMILLA)
    interfaz["graficar_metricas"](interfaz["resumir"](detalle)).savefig(FIGURAS / "metricas.pdf")
    matrices = interfaz["matrices_confusion"](detalle)
    interfaz["graficar_matrices"](matrices).savefig(FIGURAS / "matrices.pdf")
    return {
        "resumen": interfaz["resumir"](detalle).round(4).to_dict("index"),
        "matrices": {modelo: matriz.tolist() for modelo, (matriz, _) in matrices.items()},
        "clases": next(iter(matrices.values()))[1],
    }


def barridos():
    """Varía un parámetro por vez, dejando los demás en su valor por defecto."""
    salida = {}
    for modelo, parametros in BARRIDOS.items():
        for parametro, valores in parametros.items():
            curvas = interfaz["barrer"](DATASET, modelo, parametro, valores, C45_BASE, RF_BASE,
                                        REPETICIONES, SEMILLA)
            nombre = f"{'c45' if modelo == 'C4.5' else 'rf'}_{parametro}"
            interfaz["graficar_barrido"](curvas, modelo, ETIQUETAS[parametro]).savefig(FIGURAS / f"{nombre}.pdf")
            salida[nombre] = curvas.round(4).to_dict("records")
    return salida


def opciones_rf():
    """Compara las opciones no numéricas de Random Forest."""
    salida = {}
    for parametro, valores in OPCIONES_RF.items():
        for valor in valores:
            params = {**RF_BASE, parametro: valor}
            detalle = interfaz["comparar"](DATASET, C45_BASE, params, REPETICIONES, SEMILLA)
            fila = interfaz["resumir"](detalle).loc["Random Forest"]
            salida[f"{parametro}={valor}"] = {"accuracy": round(fila["Accuracy"], 4),
                                              "desvio": round(fila["Accuracy (desvío)"], 4)}
    return salida


def tamano_arbol():
    """Mide el árbol sin límites: explica por qué memoriza el entrenamiento."""
    tamanos = []
    for numero in range(REPETICIONES):
        arbol = interfaz["entrenar_arbol"](DATASET, C45_BASE, SEMILLA + numero)
        tamanos.append(interfaz["medir_arbol"](arbol.raiz_))
    nodos, hojas, profundidad = zip(*tamanos)
    return {"nodos": sum(nodos) / len(nodos), "hojas": sum(hojas) / len(hojas),
            "profundidad": sum(profundidad) / len(profundidad)}


def efecto_filtro():
    """Mide el árbol con y sin las reglas de Quinlan, para justificar su uso."""
    salida = {}
    for activo in (True, False):
        for nombre, profundidad in (("sin_limite", None), ("un_nivel", 1)):
            params = {**C45_BASE, "filtro_ganancia": activo, "max_depth": profundidad}
            detalle = interfaz["comparar"](DATASET, params, RF_BASE, REPETICIONES, SEMILLA)
            fila = interfaz["resumir"](detalle).loc["C4.5"]
            salida[f"filtro={activo},{nombre}"] = {"accuracy": round(fila["Accuracy"], 4),
                                                   "desvio": round(fila["Accuracy (desvío)"], 4)}
    return salida


if __name__ == "__main__":
    FIGURAS.mkdir(exist_ok=True)
    resultados = {"repeticiones": REPETICIONES, "semilla": SEMILLA, "base": resumen_base(),
                  "barridos": barridos(), "opciones_rf": opciones_rf(), "arbol": tamano_arbol(),
                  "filtro": efecto_filtro()}
    destino = RAIZ / "paper/resultados.json"
    destino.write_text(json.dumps(resultados, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Resultados en {destino}")
