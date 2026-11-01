"""Pruebas del código REAL del notebook, sin copiar la implementación.

Ejecutar: python -m unittest discover -s tests -p 'test_c45.py' -v
"""
import json
from pathlib import Path
import sys
import types
import unittest

import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks/01_decision_tree_c45.ipynb"
modulo = types.ModuleType("c45_desde_notebook")
sys.modules[modulo.__name__] = modulo
for celda in json.loads(NOTEBOOK.read_text())["cells"]:
    if "c45-definiciones" in celda.get("metadata", {}).get("tags", []):
        exec(compile("".join(celda["source"]), str(NOTEBOOK), "exec"), modulo.__dict__)

ArbolC45 = modulo.ArbolC45


def objetivo(clases, index=None):
    return pd.DataFrame({"objetivo": clases}, index=index)


class TestMetricas(unittest.TestCase):
    def test_entropias_manuales(self):
        self.assertEqual(modulo.entropia([]), 0)
        self.assertEqual(modulo.entropia(["A"] * 4), 0)
        self.assertEqual(modulo.entropia(["A", "B"]), 1)
        self.assertAlmostEqual(modulo.entropia(["A"] + ["B"] * 4), 0.7219280948873623)
        self.assertAlmostEqual(modulo.entropia(["A", "B", "C"]), 1.584962500721156)

    def test_ejemplo_manual_own_house(self):
        etiquetas = ["No"] * 6 + ["Yes"] * 9
        grupos = [["Yes"] * 6, ["No"] * 6 + ["Yes"] * 3]
        self.assertAlmostEqual(modulo.entropia_ponderada(grupos), 0.5509775004326937)
        self.assertAlmostEqual(modulo.ganancia_informacion(etiquetas, grupos), 0.4199730940219749)
        self.assertAlmostEqual(modulo.split_info(grupos), 0.9709505944546686)
        self.assertAlmostEqual(modulo.gain_ratio(etiquetas, grupos), 0.4325380677663126)

    def test_denominador_cero_no_es_particion_elegible(self):
        etiquetas = ["A", "B"]
        self.assertEqual(modulo.split_info([etiquetas]), 0)
        self.assertEqual(modulo.gain_ratio(etiquetas, [etiquetas]), 0)
        self.assertIsNone(modulo.particion_categorica(["x", "x"], etiquetas))

    def test_corte_perfecto_y_repetidos(self):
        corte = modulo.buscar_mejor_umbral([3, 1, 2, 2, 4], ["B", "A", "A", "A", "B"])
        self.assertEqual(corte["umbral"], 2)
        self.assertEqual(corte["ramas"], {"<=": [1, 2, 3], ">": [0, 4]})
        self.assertAlmostEqual(corte["gain_ratio"], 1)

    def test_sin_cortes_y_longitudes_incompatibles(self):
        self.assertIsNone(modulo.buscar_mejor_umbral([2, 2], ["A", "B"]))
        self.assertIsNone(modulo.buscar_mejor_umbral([], []))
        with self.assertRaises(ValueError):
            modulo.buscar_mejor_umbral([1, 2], ["A"])

    def test_busqueda_rechaza_numeros_invalidos(self):
        for valor in (float("inf"), float("nan"), "1", True, complex(1, 1)):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                modulo.buscar_mejor_umbral([valor, 2], ["A", "B"])


class TestConstruccion(unittest.TestCase):
    def test_hoja_pura(self):
        arbol = ArbolC45().fit(pd.DataFrame({"a": [1, 2]}), objetivo(["A", "A"]))
        self.assertTrue(arbol.raiz_.es_hoja)
        self.assertEqual(arbol.raiz_.motivo_parada, "clase_pura")

    def test_atributos_constantes_mayoria_y_empate(self):
        for clases, esperado in ((["A", "B", "B"], "B"), (["B", "A"], "B")):
            X = pd.DataFrame({"numero": [5] * len(clases), "categoria": ["x"] * len(clases)})
            arbol = ArbolC45().fit(X, objetivo(clases))
            self.assertTrue(arbol.raiz_.es_hoja)
            self.assertEqual(arbol.raiz_.clase, esperado)
            self.assertEqual(arbol.raiz_.motivo_parada, "sin_particion_valida")

    def test_agotamiento_categorico(self):
        X = pd.DataFrame({"a": ["x", "x", "y", "y"]})
        arbol = ArbolC45().fit(X, objetivo(["A", "B", "A", "B"]))
        self.assertEqual(set(arbol.raiz_.hijos), {"x", "y"})
        for hoja in arbol.raiz_.hijos.values():
            self.assertTrue(hoja.es_hoja)
            self.assertEqual(hoja.motivo_parada, "sin_atributos")

    def test_tres_ramas_categoricas(self):
        X = pd.DataFrame({"a": ["x", "y", "z", "x", "y", "z"]})
        y = objetivo(["A", "B", "C", "A", "B", "C"])
        arbol = ArbolC45().fit(X, y)
        self.assertEqual(len(arbol.raiz_.hijos), 3)
        self.assertEqual(arbol.predict(X)["prediccion"].tolist(), y["objetivo"].tolist())

    def test_gain_ratio_penaliza_muchos_valores(self):
        # Ambos atributos tienen Gain=1; Gain Ratio debe preferir las dos ramas.
        X = pd.DataFrame({"muchos_valores": list("abcdefgh"), "grupo": ["x"] * 4 + ["y"] * 4})
        arbol = ArbolC45().fit(X, objetivo(["A"] * 4 + ["B"] * 4))
        self.assertEqual(arbol.raiz_.atributo, "grupo")

    def test_xor_ganancia_cero_y_prepoda(self):
        X = pd.DataFrame({"a": ["0", "0", "1", "1"], "b": ["0", "1", "0", "1"]})
        y = objetivo(["A", "B", "B", "A"])
        arbol = ArbolC45(threshold=0).fit(X, y)
        self.assertEqual(arbol.raiz_.ganancia, 0)
        self.assertEqual(arbol.raiz_.atributo, "a")  # Empate: orden de columnas.
        self.assertEqual(arbol.predict(X)["prediccion"].tolist(), y["objetivo"].tolist())
        self.assertTrue(ArbolC45(threshold=0.01).fit(X, y).raiz_.es_hoja)

    def test_threshold_desigualdad_estricta(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4]})
        y = objetivo(["A", "A", "B", "B"])
        self.assertFalse(ArbolC45(threshold=1).fit(X, y).raiz_.es_hoja)
        arbol = ArbolC45(threshold=1.01).fit(X, y)
        self.assertTrue(arbol.raiz_.es_hoja)
        self.assertEqual(arbol.raiz_.motivo_parada, "ganancia_menor_que_threshold")

    def test_numero_reutilizado_y_limites(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]})
        arbol = ArbolC45().fit(X, objetivo(["A", "A", "B", "B", "A", "A"]))
        self.assertEqual(arbol.raiz_.umbral, 2)
        self.assertEqual(arbol.raiz_.hijos[">"].atributo, "x")
        self.assertEqual(arbol.raiz_.hijos[">"].umbral, 4)
        entrada = pd.DataFrame({"x": [-100, 2, 2.5, 4, 4.5, 100]})
        self.assertEqual(arbol.predict(entrada)["prediccion"].tolist(), ["A", "A", "B", "B", "A", "A"])

    def test_mixto_compara_ambos_tipos(self):
        X = pd.DataFrame({"categoria": ["r", "s", "r", "s"], "numero": [1, 2, 3, 4]})
        for etiquetas, atributo in ((["A", "B", "A", "B"], "categoria"),
                                    (["A", "A", "B", "B"], "numero")):
            with self.subTest(atributo=atributo):
                arbol = ArbolC45().fit(X, objetivo(etiquetas))
                self.assertEqual(arbol.raiz_.atributo, atributo)
                self.assertEqual(arbol.predict(X)["prediccion"].tolist(), etiquetas)

    def test_categoria_no_vista_usa_mayoria_local(self):
        X = pd.DataFrame({"grupo": ["a"] * 6 + ["b"] * 4,
                          "detalle": ["r", "s"] * 3 + ["r", "r", "r", "s"]})
        arbol = ArbolC45().fit(X, objetivo(["A"] * 6 + ["B", "B", "B", "A"]))
        self.assertEqual(arbol.raiz_.atributo, "grupo")
        consulta = pd.DataFrame({"grupo": ["b", "nuevo"], "detalle": ["nuevo", "r"]})
        self.assertEqual(arbol.predict(consulta)["prediccion"].tolist(), ["B", "A"])

    def test_categorias_numericas_declaradas_y_booleanos(self):
        X = pd.DataFrame({"categoria": pd.Categorical([1, 2, 3]), "bandera": [True, False, True]})
        arbol = ArbolC45().fit(X, objetivo(["A", "B", "C"]))
        self.assertEqual(arbol.tipos_, {"categoria": "categorico", "bandera": "categorico"})
        self.assertEqual(len(arbol.raiz_.hijos), 3)

    def test_enteros_grandes_no_pierden_precision(self):
        X = pd.DataFrame({"entero": [2**53, 2**53 + 1], "real": [0.1, 0.2]})
        arbol = ArbolC45().fit(X, objetivo(["A", "B"]))
        self.assertEqual(arbol.raiz_.atributo, "entero")
        self.assertEqual(arbol.predict(X)["prediccion"].tolist(), ["A", "B"])


class TestContrato(unittest.TestCase):
    def setUp(self):
        self.X = pd.DataFrame({"x": [1, 2, 3, 4], "c": ["r", "r", "s", "s"]}, index=[20, 4, 50, 9])
        self.y = objetivo(["A", "A", "B", "B"], self.X.index)
        self.arbol = ArbolC45().fit(self.X, self.y)

    def test_indice_orden_duplicados_y_no_mutacion(self):
        X, y = self.X.copy(deep=True), self.y.copy(deep=True)
        self.arbol.fit(X, y)
        pd.testing.assert_frame_equal(X, self.X)
        pd.testing.assert_frame_equal(y, self.y)
        consulta = X.iloc[[3, 0, 3]].copy()
        original = consulta.copy(deep=True)
        raiz = self.arbol.raiz_
        salida = self.arbol.predict(consulta)
        esperado = pd.DataFrame({"prediccion": ["B", "A", "B"]}, index=consulta.index)
        pd.testing.assert_frame_equal(salida, esperado)
        pd.testing.assert_frame_equal(consulta, original)
        self.assertIs(self.arbol.raiz_, raiz)

    def test_predict_entradas_invalidas(self):
        with self.assertRaises(TypeError):
            self.arbol.predict([[1, "r"]])
        invalidas = [self.X.iloc[:0], self.X[["c", "x"]], self.X.drop(columns="c"),
                     self.X.assign(extra=0), self.X.assign(x=float("nan")),
                     self.X.assign(x=float("inf")), self.X.assign(x="texto")]
        for X in invalidas:
            with self.subTest(columnas=list(X.columns)), self.assertRaises(ValueError):
                self.arbol.predict(X)

    def test_fit_entradas_invalidas(self):
        with self.assertRaises(TypeError):
            ArbolC45().fit(self.X.to_numpy(), self.y)
        with self.assertRaises(TypeError):
            ArbolC45().fit(self.X, self.y["objetivo"])
        for y in (self.y.iloc[::-1], self.y.rename(columns={"objetivo": "clase"}),
                  self.y.assign(extra=0), self.y.assign(objetivo=None)):
            with self.subTest(y=y.columns.tolist()), self.assertRaises(ValueError):
                ArbolC45().fit(self.X, y)
        for X in (self.X.iloc[:0], self.X.assign(x=float("inf")),
                  self.X.assign(c=None), self.X.assign(x=complex(1, 2)),
                  self.X.rename(columns={"c": "x"})):
            with self.subTest(columnas=X.columns.tolist()), self.assertRaises(ValueError):
                ArbolC45().fit(X, self.y)

    def test_threshold_invalido(self):
        for valor in (-1, float("nan"), float("inf"), "0.1", True):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                ArbolC45(threshold=valor)

    def test_predict_sin_entrenamiento(self):
        with self.assertRaises(RuntimeError):
            ArbolC45().predict(self.X)

    def test_reentrenamiento_reemplaza_arbol_y_columnas(self):
        X = pd.DataFrame({"otro": ["a", "b"]})
        self.assertIs(self.arbol.fit(X, objetivo(["C", "D"])), self.arbol)
        self.assertEqual(self.arbol.predict(X)["prediccion"].tolist(), ["C", "D"])
        with self.assertRaises(ValueError):
            self.arbol.predict(self.X)


if __name__ == "__main__":
    unittest.main()
