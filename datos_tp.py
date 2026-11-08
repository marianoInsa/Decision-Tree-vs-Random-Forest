"""Preparación común de los datos de clasificación y sus particiones.

Scikit-learn se usa aquí para cargar Iris y separar train/test, no para construir C4.5.
Se conservan las etiquetas de texto, los índices y los faltantes naturales.
"""
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Ubicación del proyecto, para encontrar los datos desde cualquier carpeta.
RAIZ = Path(__file__).resolve().parent
# Semilla fija: la misma partición se repite y los resultados se pueden reproducir.
SEMILLA = 42
FRACCION_TEST = 0.20

# Huella de cada archivo original: si un archivo cambia por error, la carga lo detecta.
HASHES = {
    "lenses.data": "711f22d5a4bed062bb715d39d23d54d2667206fbd97c94c7fd013ede47fb96c2",
    "processed.cleveland.data": "a74b7efa387bc9d108d7d0115d831fe9b414b29ae7124f331b622b4efa0427c8",
}


# Lee un archivo original comprobando primero su huella.
def leer_original(nombre, columnas, **opciones):
    contenido = (RAIZ / "data/raw" / nombre).read_bytes()
    if sha256(contenido).hexdigest() != HASHES[nombre]:
        raise ValueError(f"El archivo {nombre} no coincide con la copia original.")
    opciones.setdefault("sep", r"\s+")
    return pd.read_csv(BytesIO(contenido), header=None, names=columnas, **opciones)


# Prepara los tres datasets en un solo lugar: árbol y bosque usan exactamente lo mismo.
def cargar_datasets():
    """Devuelve {nombre: {X: DataFrame, y: DataFrame, tipo: str}}."""
    # Contact Lenses: los números se pasan a texto porque son categorías, no medidas.
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

    # Iris: viene incluido en scikit-learn y no tiene faltantes.
    iris = load_iris(as_frame=True)
    X_iris = iris.data.copy()
    y_iris = iris.target.map(dict(enumerate(iris.target_names))).to_frame("objetivo")

    # Heart Disease (Cleveland): "?" marca un faltante y 1-4 indican presencia.
    heart = leer_original("processed.cleveland.data", [
        "edad", "sexo", "dolor_pecho", "presion_reposo", "colesterol",
        "glucemia_alta", "ecg", "frecuencia_maxima", "angina_ejercicio",
        "depresion_st", "pendiente_st", "vasos", "thal", "objetivo"
    ], sep=",", na_values="?")
    X_heart = heart.drop(columns="objetivo").copy()
    # Estos códigos son categorías: tratarlos como números inventaría un orden que no existe.
    categoricas_heart = [
        "sexo", "dolor_pecho", "glucemia_alta", "ecg", "angina_ejercicio",
        "pendiente_st", "thal"
    ]
    for columna in categoricas_heart:
        # Se conserva el faltante: convertirlo en texto crearía una categoría "nan" falsa.
        X_heart[columna] = X_heart[columna].map(
            lambda valor: valor if pd.isna(valor) else str(int(valor))
        )
    # En el archivo, 0 es sano y 1 a 4 son grados de enfermedad: se agrupan en dos clases.
    y_heart = heart["objetivo"].gt(0).map(
        {False: "ausencia", True: "presencia"}
    ).to_frame("objetivo")

    return {
        "lenses": {"X": X_lenses, "y": y_lenses, "tipo": "discreto"},
        "iris": {"X": X_iris, "y": y_iris, "tipo": "continuo"},
        "heart": {"X": X_heart, "y": y_heart, "tipo": "mixto"},
    }


# Tamaños y clases esperados: los notebooks los usan para detectar datos incorrectos.
TAMANOS = {"lenses": (24, 4), "iris": (150, 4), "heart": (303, 13)}
CLASES = {
    "lenses": {"rigidos": 4, "blandos": 5, "sin_lentes": 15},
    "iris": {"setosa": 50, "versicolor": 50, "virginica": 50},
    "heart": {"ausencia": 164, "presencia": 139},
}


# stratify conserva la proporción de clases; la semilla hace repetible la partición.
def particionar(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=FRACCION_TEST, random_state=SEMILLA,
        shuffle=True, stratify=y["objetivo"]
    )
    return dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
