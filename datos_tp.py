"""Carga y partición compartidas por los notebooks 01 y 02.

Scikit-learn se usa aquí para cargar Iris y separar train/test, no para construir C4.5.
Las copias originales, etiquetas y particiones provienen de la HU de Random Forest.
"""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

RAIZ = Path(__file__).resolve().parent
SEMILLA = 42
FRACCION_TEST = 0.20

HASHES = {
    "lenses.data": "711f22d5a4bed062bb715d39d23d54d2667206fbd97c94c7fd013ede47fb96c2",
    "heart.dat": "f5f3b4204c285bafadd85cb735f38b47689f2be7047feb172dcbeab648110bf9",
}


def leer_original(nombre, columnas):
    contenido = (RAIZ / "data/raw" / nombre).read_bytes()
    if sha256(contenido).hexdigest() != HASHES[nombre]:
        raise ValueError(f"El archivo {nombre} no coincide con la copia original.")
    return pd.read_csv(BytesIO(contenido), sep=r"\s+", header=None, names=columnas)


def cargar_datasets():
    """Devuelve {nombre: {X: DataFrame, y: DataFrame, tipo: str}}."""
    lenses = leer_original("lenses.data", [
        "id", "edad", "prescripcion", "astigmatismo", "lagrimas", "objetivo"
    ]).set_index("id")
    categorias_lenses = {
        "edad": {1: "joven", 2: "prepresbicia", 3: "presbicia"},
        "prescripcion": {1: "miopia", 2: "hipermetropia"},
        "astigmatismo": {1: "no", 2: "si"},
        "lagrimas": {1: "reducida", 2: "normal"},
    }
    X_lenses = lenses.drop(columns="objetivo").copy()
    for columna, etiquetas in categorias_lenses.items():
        X_lenses[columna] = X_lenses[columna].map(etiquetas)
    y_lenses = lenses[["objetivo"]].copy()
    y_lenses["objetivo"] = y_lenses["objetivo"].map(
        {1: "rigidos", 2: "blandos", 3: "sin_lentes"}
    )

    iris = load_iris(as_frame=True)
    X_iris = iris.data.copy()
    y_iris = iris.target.map(dict(enumerate(iris.target_names))).to_frame("objetivo")

    heart = leer_original("heart.dat", [
        "edad", "sexo", "dolor_pecho", "presion_reposo", "colesterol",
        "glucemia_alta", "ecg", "frecuencia_maxima", "angina_ejercicio",
        "depresion_st", "pendiente_st", "vasos", "thal", "objetivo"
    ])
    X_heart = heart.drop(columns="objetivo").copy()
    categoricas_heart = [
        "sexo", "dolor_pecho", "glucemia_alta", "ecg", "angina_ejercicio",
        "pendiente_st", "thal"
    ]
    for columna in categoricas_heart:
        X_heart[columna] = X_heart[columna].astype(int).astype(str)
    y_heart = heart[["objetivo"]].copy()
    y_heart["objetivo"] = y_heart["objetivo"].map({1: "ausencia", 2: "presencia"})

    return {
        "lenses": {"X": X_lenses, "y": y_lenses, "tipo": "discreto"},
        "iris": {"X": X_iris, "y": y_iris, "tipo": "continuo"},
        "heart": {"X": X_heart, "y": y_heart, "tipo": "mixto"},
    }


TAMANOS = {"lenses": (24, 4), "iris": (150, 4), "heart": (270, 13)}
CLASES = {
    "lenses": {"rigidos": 4, "blandos": 5, "sin_lentes": 15},
    "iris": {"setosa": 50, "versicolor": 50, "virginica": 50},
    "heart": {"ausencia": 150, "presencia": 120},
}


def particionar(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=FRACCION_TEST, random_state=SEMILLA,
        shuffle=True, stratify=y["objetivo"]
    )
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
