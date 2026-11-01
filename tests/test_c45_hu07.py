"""HU-07: controles de crecimiento del C4.5 real del notebook.

Carga las celdas de definiciones con `test_c45`, igual que la suite de HU-08;
no mantiene una segunda implementación. Ejecutar:

    uv run --locked python -m unittest discover -s tests -p 'test_c45*.py' -v
"""
import unittest

import pandas as pd

import test_c45 as base


ArbolC45 = base.ArbolC45
objetivo = base.objetivo
modulo = base.modulo


def medir_arbol(nodo, profundidad=0):
    """Devuelve (profundidad máxima, cantidad de hojas) de forma recursiva."""
    if nodo.es_hoja:
        return profundidad, 1
    medidas = [medir_arbol(hijo, profundidad + 1) for hijo in nodo.hijos.values()]
    return max(prof for prof, _ in medidas), sum(hojas for _, hojas in medidas)


class TestValidacionControles(unittest.TestCase):
    def test_defaults_y_compatibilidad_posicional(self):
        arbol = ArbolC45()
        self.assertEqual(arbol.threshold, 0)
        self.assertIsNone(arbol.max_depth)
        self.assertEqual(arbol.min_samples_split, 2)
        self.assertEqual(arbol.min_gain_ratio, 0)

        # El primer argumento posicional sigue siendo threshold.
        viejo_estilo = ArbolC45(0.25)
        self.assertEqual(viejo_estilo.threshold, 0.25)
        self.assertIsNone(viejo_estilo.max_depth)

        completo = ArbolC45(0.25, 3, 5, 0.1)
        self.assertEqual(
            (completo.threshold, completo.max_depth,
             completo.min_samples_split, completo.min_gain_ratio),
            (0.25, 3, 5, 0.1),
        )

    def test_max_depth_invalido(self):
        for valor in (-1, 1.5, 2.0, True, "1"):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                ArbolC45(max_depth=valor)
        for valor in (None, 0, 1, 5):
            self.assertEqual(ArbolC45(max_depth=valor).max_depth, valor)

    def test_min_samples_split_invalido(self):
        for valor in (1, 1.99, -1, True, None, float("nan"), float("inf"), "2"):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                ArbolC45(min_samples_split=valor)
        for valor in (2, 2.0, 2.5):
            self.assertEqual(ArbolC45(min_samples_split=valor).min_samples_split, valor)

    def test_min_gain_ratio_invalido(self):
        for valor in (-0.01, True, None, float("nan"), float("inf"), "0"):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                ArbolC45(min_gain_ratio=valor)
        for valor in (0, 0.0, 0.5):
            self.assertEqual(ArbolC45(min_gain_ratio=valor).min_gain_ratio, valor)

    def test_fit_revalida_parametros_modificados(self):
        X = pd.DataFrame({"x": [1, 2]})
        y = objetivo(["A", "B"])
        for atributo, valor in (("max_depth", -1),
                                ("min_samples_split", 1),
                                ("min_gain_ratio", -0.1)):
            with self.subTest(atributo=atributo):
                arbol = ArbolC45()
                setattr(arbol, atributo, valor)
                with self.assertRaises(ValueError):
                    arbol.fit(X, y)


class TestMaxDepth(unittest.TestCase):
    def setUp(self):
        # El árbol completo mide 2 niveles y deja 3 hojas.
        self.X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]})
        self.y = objetivo(["A", "A", "B", "B", "A", "A"])

    def test_cero_uno_dos_y_none(self):
        esperado = {0: (0, 1), 1: (1, 2), 2: (2, 3), None: (2, 3)}
        for max_depth, medidas in esperado.items():
            with self.subTest(max_depth=max_depth):
                arbol = ArbolC45(max_depth=max_depth).fit(self.X, self.y)
                self.assertEqual(medir_arbol(arbol.raiz_), medidas)

    def test_hoja_por_profundidad_conserva_distribucion_y_clase(self):
        arbol = ArbolC45(max_depth=0).fit(self.X, self.y)
        raiz = arbol.raiz_
        self.assertTrue(raiz.es_hoja)
        self.assertEqual(raiz.motivo_parada, "profundidad_maxima")
        self.assertEqual(raiz.conteos, {"A": 4, "B": 2})
        self.assertEqual(raiz.clase, "A")
        self.assertEqual(raiz.peso_total, 6)
        p = arbol.predict_proba(self.X)
        self.assertEqual(p.columns.tolist(), ["A", "B"])
        for fila in p.itertuples(index=False):
            self.assertAlmostEqual(fila.A, 4 / 6)
            self.assertAlmostEqual(fila.B, 2 / 6)

    def test_max_depth_detiene_antes_de_evaluar_candidatos(self):
        X = pd.DataFrame({"g": ["a", "a", "b", "b"],
                          "h": ["u", "v", "u", "v"]})
        y = objetivo(["A", "B", "B", "A"])
        uno = ArbolC45(max_depth=1).fit(X, y)
        self.assertEqual(medir_arbol(uno.raiz_), (1, 2))
        for hijo in uno.raiz_.hijos.values():
            self.assertEqual(hijo.motivo_parada, "profundidad_maxima")
        # Con un nivel más, esos mismos hijos sí encuentran una partición.
        dos = ArbolC45(max_depth=2).fit(X, y)
        self.assertEqual(medir_arbol(dos.raiz_), (2, 4))


class TestMinSamplesSplit(unittest.TestCase):
    def setUp(self):
        # Cada hijo de g tiene exactamente masa 2 y puede dividirse por h.
        self.X = pd.DataFrame({"g": ["a", "a", "b", "b"],
                               "h": ["u", "v", "u", "v"]})
        self.y = objetivo(["A", "B", "B", "A"])

    def test_igualdad_permite_dividir_y_valor_mayor_bloquea(self):
        exacto = ArbolC45(min_samples_split=2).fit(self.X, self.y)
        self.assertEqual(medir_arbol(exacto.raiz_), (2, 4))
        for hijo in exacto.raiz_.hijos.values():
            self.assertFalse(hijo.es_hoja)

        arriba = ArbolC45(min_samples_split=2.01).fit(self.X, self.y)
        self.assertEqual(medir_arbol(arriba.raiz_), (1, 2))
        for hijo in arriba.raiz_.hijos.values():
            self.assertTrue(hijo.es_hoja)
            self.assertEqual(hijo.motivo_parada, "masa_menor_que_min_samples_split")

    def test_masa_efectiva_no_cuenta_filas_de_copias_faltantes(self):
        X = pd.DataFrame({"g": ["a", "b", None], "h": ["u", "u", "v"]})
        y = objetivo(["A", "B", "A"])
        arbol = ArbolC45().fit(X, y)
        self.assertEqual(arbol.raiz_.atributo, "g")
        self.assertEqual(arbol.raiz_.conteos, {"A": 2, "B": 1})

        # La copia faltante crea una segunda fila, pero solo aporta peso 0.5.
        hijo = arbol.raiz_.hijos["b"]
        self.assertEqual(hijo.n_ejemplos, 2)
        self.assertAlmostEqual(hijo.peso_total, 1.5)
        self.assertEqual(hijo.conteos, {"A": 0.5, "B": 1.0})
        self.assertTrue(hijo.es_hoja)
        self.assertEqual(hijo.motivo_parada, "masa_menor_que_min_samples_split")

        # El corte sería válido: lo bloquea la masa, no la cantidad de filas.
        candidata = modulo.particion_categorica(["u", "v"], ["B", "A"], [1.0, 0.5])
        self.assertIsNotNone(candidata)
        self.assertGreater(candidata["gain_ratio"], 0)

        # La hoja conserva la distribución ponderada y la mezcla con faltantes.
        consulta = pd.DataFrame({"g": ["b", "b", None],
                                 "h": ["u", "v", "u"]})
        p = arbol.predict_proba(consulta)
        self.assertAlmostEqual(p.iloc[0]["A"], 1 / 3)
        self.assertAlmostEqual(p.iloc[0]["B"], 2 / 3)
        self.assertAlmostEqual(p.iloc[1]["A"], 1 / 3)
        self.assertAlmostEqual(p.iloc[2]["A"], 2 / 3)
        self.assertAlmostEqual(p.iloc[2]["B"], 1 / 3)


class TestMinGainRatio(unittest.TestCase):
    def test_igualdad_permite_dividir_y_valor_mayor_bloquea(self):
        # Gain=1 bit; SplitInfo=2; Gain Ratio=0.5 exacto.
        X = pd.DataFrame({"g": ["a", "b", "c", "d"]})
        y = objetivo(["A", "B", "A", "B"])

        exacto = ArbolC45(min_gain_ratio=0.5).fit(X, y)
        self.assertFalse(exacto.raiz_.es_hoja)
        self.assertAlmostEqual(exacto.raiz_.gain_ratio, 0.5)
        self.assertEqual(medir_arbol(exacto.raiz_), (1, 4))

        arriba = ArbolC45(min_gain_ratio=0.500001).fit(X, y)
        self.assertTrue(arriba.raiz_.es_hoja)
        self.assertEqual(arriba.raiz_.motivo_parada,
                         "gain_ratio_menor_que_min_gain_ratio")

    def test_threshold_bits_y_ratio_adimensional_son_independientes(self):
        X = pd.DataFrame({"g": ["a", "b", "c", "d"]})
        y = objetivo(["A", "B", "A", "B"])

        # La ganancia (1 bit) pasa threshold=0.75; el ratio (0.5) todavía no.
        self.assertFalse(ArbolC45(threshold=0.75, min_gain_ratio=0).fit(X, y).raiz_.es_hoja)

        bloqueado_por_ratio = ArbolC45(threshold=0.75, min_gain_ratio=0.75).fit(X, y)
        self.assertTrue(bloqueado_por_ratio.raiz_.es_hoja)
        self.assertEqual(bloqueado_por_ratio.raiz_.motivo_parada,
                         "gain_ratio_menor_que_min_gain_ratio")

        # threshold sigue bloqueando por sí solo, con min_gain_ratio=0.
        bloqueado_por_bits = ArbolC45(threshold=1.01, min_gain_ratio=0).fit(X, y)
        self.assertTrue(bloqueado_por_bits.raiz_.es_hoja)
        self.assertEqual(bloqueado_por_bits.raiz_.motivo_parada,
                         "ganancia_menor_que_threshold")


class TestRegresionSinPodaPorDefecto(unittest.TestCase):
    def test_xor_ganancia_cero_y_ratio_cero_siguen_dividiendo(self):
        X = pd.DataFrame({"a": ["0", "0", "1", "1"],
                          "b": ["0", "1", "0", "1"]})
        y = objetivo(["No", "Yes", "Yes", "No"])
        arbol = ArbolC45().fit(X, y)
        self.assertEqual(arbol.raiz_.ganancia, 0)
        self.assertEqual(arbol.raiz_.gain_ratio, 0)
        self.assertEqual(medir_arbol(arbol.raiz_), (2, 4))
        self.assertEqual(arbol.predict(X)["prediccion"].tolist(),
                         y["objetivo"].tolist())

    def test_fixture_numerico_reutiliza_atributo_y_mantiene_hojas(self):
        X = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6]})
        y = objetivo(["A", "A", "B", "B", "A", "A"])
        arbol = ArbolC45().fit(X, y)
        self.assertEqual(arbol.raiz_.umbral, 2)
        self.assertEqual(arbol.raiz_.hijos[">"].atributo, "x")
        self.assertEqual(medir_arbol(arbol.raiz_), (2, 3))
        self.assertEqual(arbol.predict(X)["prediccion"].tolist(),
                         y["objetivo"].tolist())

    def test_fixture_categorico_conserva_tres_ramas(self):
        X = pd.DataFrame({"a": ["x", "y", "z", "x", "y", "z"]})
        y = objetivo(["A", "B", "C", "A", "B", "C"])
        arbol = ArbolC45().fit(X, y)
        self.assertEqual(set(arbol.raiz_.hijos), {"x", "y", "z"})
        self.assertEqual(medir_arbol(arbol.raiz_), (1, 3))


if __name__ == "__main__":
    unittest.main()
