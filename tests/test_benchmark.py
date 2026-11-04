"""Pruebas del motor real de 03_experimentos.ipynb."""

import json
from pathlib import Path
import unittest

import pandas as pd
from sklearn.model_selection import train_test_split


RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks/03_experimentos.ipynb"
espacio = {}
for celda in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
    if "benchmark-definiciones" in celda.get("metadata", {}).get("tags", []):
        exec(compile("".join(celda["source"]), str(NOTEBOOK), "exec"), espacio)

ejecutar_benchmark = espacio["ejecutar_benchmark"]


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from datos_tp import cargar_datasets, particionar

        datos = cargar_datasets()["iris"]
        partes = particionar(datos["X"], datos["y"])
        cls.X_train, cls.X_valid, cls.y_train, cls.y_valid = train_test_split(
            partes["X_train"], partes["y_train"], test_size=0.25,
            random_state=42, stratify=partes["y_train"]["objetivo"])
        cls.grillas = {
            "C4.5": {"max_depth": [1, 3]},
            "Random Forest": {"n_estimators": [10, 20]},
        }

    def ejecutar(self, grillas=None, **kwargs):
        return ejecutar_benchmark(
            "iris", self.X_train, self.y_train, self.X_valid, self.y_valid,
            self.grillas if grillas is None else grillas, semilla=42, **kwargs)

    def test_grillas_metricas_y_esquema(self):
        salida = self.ejecutar()
        self.assertEqual(len(salida), 8)
        self.assertEqual(salida.groupby("modelo")["ejecucion"].nunique().to_dict(),
                         {"C4.5": 2, "Random Forest": 2})
        self.assertEqual(set(salida["conjunto"]), {"entrenamiento", "validacion"})
        self.assertEqual(set(salida["n_casos"]), {len(self.X_train), len(self.X_valid)})
        self.assertEqual(set(salida["semilla"]), {42})
        self.assertEqual(set(salida["protocolo_particion"]), {"particiones provistas"})
        self.assertEqual(set(salida["precision_macro"].between(0, 1)), {True})
        self.assertEqual(set(salida["recall_weighted"].between(0, 1)), {True})
        self.assertTrue((salida["tiempo_entrenamiento_ms"] > 0).all())
        self.assertTrue((salida["tiempo_inferencia_ms"] > 0).all())
        for fila in salida.itertuples():
            self.assertEqual(sum(map(sum, fila.matriz_confusion)), fila.n_casos)
            self.assertEqual(len(fila.clases), len(fila.matriz_confusion))
            self.assertAlmostEqual(fila.recall_weighted, fila.accuracy)
            self.assertEqual(fila.parametros_efectivos["max_depth"],
                             fila.parametros.get("max_depth", None))

    def test_parametrizacion_rf_entrena_y_predice(self):
        modelo = espacio["rf"].crear_modelo(
            self.X_train, n_estimators=7, max_features=None, max_depth=2,
            criterion="entropy", bootstrap=False, random_state=13)
        parametros = modelo.named_steps["random_forest"].get_params()
        for nombre, valor in {"n_estimators": 7, "max_features": None,
                              "max_depth": 2, "criterion": "entropy",
                              "bootstrap": False, "random_state": 13}.items():
            self.assertEqual(parametros[nombre], valor)
        modelo.fit(self.X_train, self.y_train["objetivo"])
        predicciones = espacio["rf"].crear_predictor(modelo)(self.X_valid)
        self.assertTrue(predicciones.index.equals(self.X_valid.index))
        self.assertEqual(list(predicciones.columns), ["prediccion"])

    def test_repeticion_preserva_predicciones_y_metricas(self):
        grillas = {"C4.5": {"max_depth": [2]},
                   "Random Forest": {"n_estimators": [10]}}
        primera = self.ejecutar(grillas)
        segunda = self.ejecutar(grillas)
        columnas = [c for c in primera if not c.startswith("tiempo_")]
        pd.testing.assert_frame_equal(primera[columnas], segunda[columnas])

    def test_parametros_invalidos_y_particiones_mal_alineadas(self):
        with self.assertRaisesRegex(ValueError, "parámetros no admitidos"):
            self.ejecutar({"C4.5": {"criterio": ["gini"]}, "Random Forest": {}})
        with self.assertRaisesRegex(ValueError, "Random Forest.*n_estimators"):
            self.ejecutar({"C4.5": {}, "Random Forest": {"n_estimators": [0]}})
        with self.assertRaisesRegex(ValueError, "índices de X e y"):
            ejecutar_benchmark("iris", self.X_train, self.y_train.reset_index(drop=True),
                               self.X_valid, self.y_valid, self.grillas)
        with self.assertRaisesRegex(ValueError, "índices disjuntos"):
            ejecutar_benchmark("iris", self.X_train, self.y_train,
                               self.X_train, self.y_train, self.grillas)


if __name__ == "__main__":
    unittest.main()
