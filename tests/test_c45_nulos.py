"""HU-08: resultados manuales, conservación de masa y contrato probabilístico.

Ejecutar: python -m unittest discover -s tests -p 'test_c45*.py' -v
"""
import math
import unittest

import pandas as pd

import test_c45 as base


ArbolC45 = base.ArbolC45
objetivo = base.objetivo
modulo = base.modulo


class TestMetricasConNulos(unittest.TestCase):
    def test_entropia_con_pesos_fraccionarios(self):
        # Masas A=1, B=3: H=0.811278..., no H de las tres filas.
        self.assertAlmostEqual(modulo.entropia(["A", "A", "B"], [0.25, 0.75, 3]),
                               0.8112781244591328)
        self.assertEqual(modulo.entropia(["A", "B"], [0, 0]), 0)

    def test_ganancia_numerica_y_split_info_manuales(self):
        corte = modulo.buscar_mejor_umbral([1, 2, None], ["A", "B", "A"])
        # Conocidos perfectamente separados: Gain=2/3; masas 1/3,1/3,1/3.
        self.assertEqual(corte["umbral"], 1)
        self.assertAlmostEqual(corte["ganancia"], 2 / 3)
        self.assertAlmostEqual(corte["split_info"], math.log2(3))
        self.assertAlmostEqual(corte["gain_ratio"], (2 / 3) / math.log2(3))
        self.assertEqual(corte["proporciones"], {"<=": 0.5, ">": 0.5})
        self.assertEqual(corte["faltantes"], [2])
        otra_clase_desconocida = modulo.buscar_mejor_umbral([1, 2, None], ["A", "B", "B"])
        self.assertEqual(corte["gain_ratio"], otra_clase_desconocida["gain_ratio"])

    def test_ganancia_categorica_manual(self):
        corte = modulo.particion_categorica(["a", "a", "b", None], ["A", "B", "B", "A"])
        conocido = 0.9182958340544896 - 2 / 3
        self.assertAlmostEqual(corte["ganancia"], 0.75 * conocido)
        self.assertAlmostEqual(corte["split_info"], 1.5)
        self.assertAlmostEqual(corte["gain_ratio"], 0.75 * conocido / 1.5)
        self.assertEqual(corte["proporciones"], {"a": 2 / 3, "b": 1 / 3})

    def test_pesos_de_conocidos_determinan_proporciones(self):
        corte = modulo.buscar_mejor_umbral([1, 2, pd.NA], ["A", "B", "A"], [1, 3, 2])
        self.assertEqual(corte["proporciones"], {"<=": 0.25, ">": 0.75})
        self.assertAlmostEqual(corte["ganancia"], 4 / 6 * 0.8112781244591328)
        self.assertAlmostEqual(corte["split_info"],
                               -(1 / 6 * math.log2(1 / 6) + 0.5 * math.log2(0.5)
                                 + 1 / 3 * math.log2(1 / 3)))

    def test_sin_conocidos_o_con_un_solo_valor_no_hay_particion(self):
        for valores in ([None, pd.NA], [1, None], [1, 1, None]):
            etiquetas = ["A", "B", "A"][:len(valores)]
            self.assertIsNone(modulo.buscar_mejor_umbral(valores, etiquetas))
            self.assertIsNone(modulo.particion_categorica(valores, etiquetas))

    def test_pesos_cero_no_proponen_cortes(self):
        self.assertIsNone(modulo.buscar_mejor_umbral([1, 2], ["A", "B"], [1, 0]))
        self.assertIsNone(modulo.particion_categorica(["a", "b"], ["A", "B"], [1, 0]))

    def test_pesos_invalidos(self):
        for pesos in ([1], [1, -1], [1, float("nan")], [1, float("inf")], [1, True]):
            with self.subTest(pesos=pesos), self.assertRaises(ValueError):
                modulo.entropia(["A", "B"], pesos)


class TestEntrenamientoConNulos(unittest.TestCase):
    def test_reparto_numerico_y_probabilidades_manuales(self):
        X = pd.DataFrame({"x": [1, 2, None]})
        arbol = ArbolC45().fit(X, objetivo(["A", "B", "A"]))
        raiz = arbol.raiz_
        self.assertEqual(raiz.peso_total, 3)
        self.assertEqual(raiz.hijos["<="].conteos, {"A": 1.5})
        self.assertEqual(raiz.hijos[">"].conteos, {"A": 0.5, "B": 1.0})
        consulta = pd.DataFrame({"x": [None, 1, 2]}, index=[8, 8, 3])
        esperado = pd.DataFrame({"A": [2 / 3, 1, 1 / 3], "B": [1 / 3, 0, 2 / 3]},
                               index=consulta.index)
        pd.testing.assert_frame_equal(arbol.predict_proba(consulta), esperado)
        self.assertEqual(arbol.predict(consulta)["prediccion"].tolist(), ["A", "A", "B"])

    def test_masa_por_clase_se_conserva_en_todos_los_niveles(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4, None, 2, None, 4],
                          "c": ["a", None, "a", "b", "b", "b", None, "a"]})
        arbol = ArbolC45().fit(X, objetivo(["A", "A", "B", "B", "A", "B", "B", "A"]))

        def verificar(nodo):
            self.assertAlmostEqual(sum(nodo.conteos.values()), nodo.peso_total)
            if not nodo.es_hoja:
                self.assertAlmostEqual(sum(nodo.proporciones.values()), 1)
                self.assertEqual(set(nodo.proporciones), set(nodo.hijos))
                for clase in arbol.classes_:
                    self.assertAlmostEqual(sum(h.conteos.get(clase, 0) for h in nodo.hijos.values()),
                                           nodo.conteos.get(clase, 0))
                for hijo in nodo.hijos.values():
                    self.assertGreater(hijo.peso_total, 0)
                    verificar(hijo)

        self.assertTrue(any(not h.es_hoja for h in arbol.raiz_.hijos.values()))
        verificar(arbol.raiz_)
        p = arbol.predict_proba(pd.DataFrame({"x": [None, 1, 3], "c": [None, None, "nueva"]}))
        self.assertTrue(((p >= 0) & (p <= 1)).all().all())
        for total in p.sum(axis=1):
            self.assertAlmostEqual(total, 1)
        # Con todos los atributos desconocidos se recupera el prior del nodo raíz.
        self.assertAlmostEqual(p.iloc[0]["A"], 0.5)
        self.assertAlmostEqual(p.iloc[0]["B"], 0.5)

    def test_columna_totalmente_faltante_se_ignora(self):
        X = pd.DataFrame({"vacio": [None] * 4, "x": [1, 2, 3, 4]})
        arbol = ArbolC45().fit(X, objetivo(["A", "A", "B", "B"]))
        self.assertEqual(arbol.raiz_.atributo, "x")

    def test_todos_los_atributos_faltantes_crean_hoja_con_prior(self):
        X = pd.DataFrame({"x": [None, float("nan"), pd.NA]})
        arbol = ArbolC45().fit(X, objetivo(["B", "A", "A"]))
        self.assertTrue(arbol.raiz_.es_hoja)
        self.assertEqual(arbol.raiz_.motivo_parada, "sin_particion_valida")
        self.assertEqual(arbol.predict(X)["prediccion"].tolist(), ["A"] * 3)
        self.assertEqual(arbol.classes_.tolist(), ["B", "A"])
        self.assertAlmostEqual(arbol.predict_proba(X).iloc[0]["A"], 2 / 3)

    def test_threshold_se_aplica_a_ganancia_ajustada(self):
        X = pd.DataFrame({"x": [1, 2, None]})
        y = objetivo(["A", "B", "A"])
        self.assertFalse(ArbolC45(threshold=2 / 3).fit(X, y).raiz_.es_hoja)
        self.assertTrue(ArbolC45(threshold=0.7).fit(X, y).raiz_.es_hoja)

    def test_etiquetas_faltantes_siguen_rechazadas(self):
        with self.assertRaises(ValueError):
            ArbolC45().fit(pd.DataFrame({"x": [1, None]}), objetivo(["A", None]))


class TestInferenciaProbabilistica(unittest.TestCase):
    def test_mezcla_distribuciones_no_votos_de_mayorias(self):
        # Rama a: P(A)=0.6, peso 0.6; rama b: P(B)=1, peso 0.4.
        # Voto de mayorías diría A; mezcla correcta da P(B)=0.64.
        X = pd.DataFrame({"c": ["a"] * 15 + ["b"] * 10})
        arbol = ArbolC45().fit(X, objetivo(["A"] * 9 + ["B"] * 16))
        p = arbol.predict_proba(pd.DataFrame({"c": [None]}))
        self.assertAlmostEqual(p.iloc[0]["A"], 0.36)
        self.assertAlmostEqual(p.iloc[0]["B"], 0.64)
        self.assertEqual(arbol.predict(pd.DataFrame({"c": [None]})).iloc[0, 0], "B")

    def test_categoria_nueva_usa_distribucion_local_sin_mezclar(self):
        X = pd.DataFrame({"grupo": ["a"] * 6 + ["b"] * 4,
                          "detalle": ["r", "s"] * 3 + ["r", "r", "r", "s"]})
        arbol = ArbolC45().fit(X, objetivo(["A"] * 6 + ["B", "B", "B", "A"]))
        consulta = pd.DataFrame({"grupo": ["b", "nuevo"], "detalle": ["nuevo", "r"]})
        esperado = pd.DataFrame({"A": [0.25, 0.7], "B": [0.75, 0.3]})
        pd.testing.assert_frame_equal(arbol.predict_proba(consulta), esperado)
        self.assertEqual(arbol.predict(consulta)["prediccion"].tolist(), ["B", "A"])

    def test_empates_globales_y_orden_de_clases(self):
        X = pd.DataFrame({"c": ["r", "s", "s", "s", "s"]})
        arbol = ArbolC45().fit(X, objetivo(["Z", "A", "Z", "A", "Z"]))
        self.assertEqual(arbol.classes_.tolist(), ["Z", "A"])
        # En s aparece A primero, pero el empate usa el orden global Z,A.
        consulta = pd.DataFrame({"c": ["s"]})
        self.assertEqual(arbol.predict(consulta).iloc[0, 0], "Z")
        self.assertEqual(arbol.predict_proba(consulta).idxmax(axis=1).iloc[0], "Z")

    def test_empate_de_atributos_y_cortes_determinista(self):
        X = pd.DataFrame({"b": [1, 2, 3, None], "a": [1, 2, 3, None]})
        arbol = ArbolC45().fit(X, objetivo(["A", "B", "A", "B"]))
        self.assertEqual(arbol.raiz_.atributo, "b")
        self.assertEqual(arbol.raiz_.umbral, 1)

    def test_tipos_nullable_y_distintas_representaciones_de_nulos(self):
        X = pd.DataFrame({"x": pd.Series([1, 2, pd.NA], dtype="Int64"),
                          "c": pd.Series(["a", "b", pd.NA], dtype="string")})
        y = objetivo(["A", "B", "A"])
        arbol = ArbolC45().fit(X, y)
        for valor in (None, float("nan"), pd.NA, pd.NaT):
            consulta = pd.DataFrame({"x": [valor], "c": [valor]})
            p = arbol.predict_proba(consulta)
            self.assertAlmostEqual(p.iloc[0].sum(), 1)
            self.assertEqual(arbol.predict(consulta).iloc[0, 0], "A")

    def test_indice_duplicado_no_mutacion_y_probabilidades_consistentes(self):
        X = pd.DataFrame({"x": [1, 2, None]}, index=[8, 8, 4])
        y = objetivo(["A", "B", "A"], X.index)
        original, original_y = X.copy(deep=True), y.copy(deep=True)
        arbol = ArbolC45().fit(X, y)
        consulta = X.iloc[[2, 0, 2]].copy()
        original_consulta = consulta.copy(deep=True)
        p = arbol.predict_proba(consulta)
        pd.testing.assert_frame_equal(X, original)
        pd.testing.assert_frame_equal(y, original_y)
        pd.testing.assert_frame_equal(consulta, original_consulta)
        self.assertTrue(p.index.equals(consulta.index))
        self.assertEqual(p.columns.tolist(), arbol.classes_.tolist())
        self.assertEqual(arbol.predict(consulta)["prediccion"].tolist(), p.idxmax(axis=1).tolist())

    def test_predict_proba_valida_igual_que_predict(self):
        X = pd.DataFrame({"x": [1, 2]})
        arbol = ArbolC45().fit(X, objetivo(["A", "B"]))
        with self.assertRaises(RuntimeError):
            ArbolC45().predict_proba(X)
        with self.assertRaises(TypeError):
            arbol.predict_proba([[1]])
        for consulta in (X.iloc[:0], X.rename(columns={"x": "otra"}), X.assign(x=float("inf")),
                         X.assign(x="texto")):
            with self.subTest(consulta=consulta), self.assertRaises(ValueError):
                arbol.predict_proba(consulta)

    def test_reentrenamiento_reemplaza_classes(self):
        arbol = ArbolC45().fit(pd.DataFrame({"x": [1, None]}), objetivo(["B", "A"]))
        arbol.fit(pd.DataFrame({"c": ["a", None]}), objetivo(["C", "C"]))
        p = arbol.predict_proba(pd.DataFrame({"c": [None]}))
        self.assertEqual(p.columns.tolist(), ["C"])
        self.assertEqual(p.iloc[0, 0], 1)


if __name__ == "__main__":
    unittest.main()
