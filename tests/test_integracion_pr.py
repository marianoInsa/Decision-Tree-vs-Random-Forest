"""Comprueba los aportes adaptados sin mantener otra implementación de los modelos."""
import json
from pathlib import Path
import unittest
import numpy as np
import pandas as pd
import test_experimentos as base

modulo = base.modulo


class TestCleveland(unittest.TestCase):
    def test_conserva_casos_clases_ausencias_y_particiones(self):
        datos = modulo.cargar_datos_experimentales()["heart"]
        atributos, objetivos = datos["caracteristicas"], datos["objetivos"]
        self.assertEqual(atributos.shape, (303, 13))
        self.assertEqual(objetivos.objetivo.value_counts().to_dict(), {"ausencia": 164, "presencia": 139})
        self.assertEqual(atributos.isna().sum()[["vasos", "thal"]].tolist(), [4, 2])
        self.assertEqual(int(atributos.isna().any(axis=1).sum()), 6)
        partes = modulo.preparar_particiones(atributos, objetivos)["referencia"]
        self.assertEqual([len(partes[nombre][0]) for nombre in ("entrenamiento", "prueba")], [242, 61])
        self.assertTrue(partes["entrenamiento"][0].index.intersection(partes["prueba"][0].index).empty)

    def test_cero_inyectado_conserva_ausencias_naturales(self):
        datos = modulo.cargar_datos_experimentales()["heart"]
        partes = modulo.preparar_particiones(datos["caracteristicas"], datos["objetivos"])["referencia"]
        cero = modulo.inyectar_faltantes(partes, 0, 42)
        for nombre in partes:
            pd.testing.assert_frame_equal(partes[nombre][0], cero[nombre][0])
            pd.testing.assert_frame_equal(partes[nombre][1], cero[nombre][1])


class TestFiltroGanancia(unittest.TestCase):
    def test_corte_por_ganancia_y_por_razon_son_distintos(self):
        # La separación en 4 reduce más incertidumbre; la de 6 tiene mayor cociente.
        clases = list("AAAABAB")
        arbol = modulo.modulo_arbol
        self.assertEqual(arbol.buscar_mejor_umbral(range(1, 8), clases)["umbral"], 6)
        self.assertEqual(arbol.buscar_mejor_umbral(range(1, 8), clases, por_ganancia=True)["umbral"], 4)

    def test_filtro_de_atributos_y_valor_inicial(self):
        atributos = pd.DataFrame({"grupo": list("aaaaaabb"), "identidad": list("abcdefgh")})
        objetivos = pd.DataFrame({"objetivo": list("AAAABBBB")})
        clase = modulo.modulo_arbol.ArbolC45
        inicial = clase().entrenar(atributos, objetivos)
        filtrado = clase(filtro_ganancia=True).entrenar(atributos, objetivos)
        self.assertFalse(inicial.obtener_parametros()["filtro_ganancia"])
        self.assertEqual(inicial.raiz_.atributo, "grupo")
        self.assertEqual(filtrado.raiz_.atributo, "identidad")

    def test_filtro_valida_tipo_y_conserva_masa_con_ausentes(self):
        clase = modulo.modulo_arbol.ArbolC45
        for valor in (1, "si", None):
            with self.subTest(valor=valor), self.assertRaises(ValueError):
                clase(filtro_ganancia=valor)
        atributos = pd.DataFrame({"numero": [1, 2, 3, None], "grupo": ["a", None, "b", "b"]})
        objetivos = pd.DataFrame({"objetivo": list("ABBA")})
        modelo = clase(filtro_ganancia=True).entrenar(atributos, objetivos)
        def revisar(nodo):
            if not nodo.es_hoja:
                for etiqueta in modelo.classes_:
                    self.assertAlmostEqual(sum(hijo.conteos.get(etiqueta, 0) for hijo in nodo.hijos.values()),
                                           nodo.conteos.get(etiqueta, 0))
                for hijo in nodo.hijos.values():
                    revisar(hijo)
        revisar(modelo.raiz_)
        probabilidades = modelo.predecir_probabilidades(atributos)
        np.testing.assert_allclose(probabilidades.sum(axis=1), 1)


class TestImputacionFrecuente(unittest.TestCase):
    def test_aprende_solo_mediana_y_moda_de_entrenamiento(self):
        atributos = pd.DataFrame({"numero": [1.0, 3.0, None], "grupo": ["a", "a", None]})
        objetivos = pd.DataFrame({"objetivo": list("ABA")})
        modelo = modulo.modulo_bosque.entrenar_bosque(atributos, objetivos,
            numero_arboles=3, tratamiento_faltantes="imputar_frecuente")
        preparador = modelo.named_steps["preprocesador"]
        self.assertEqual(preparador.named_transformers_["numericas"].statistics_.tolist(), [2])
        imputador = preparador.named_transformers_["categoricas"].named_steps["imputar"]
        self.assertEqual(imputador.statistics_.tolist(), ["a"])
        consulta = pd.DataFrame({"numero": [1000.0, None], "grupo": ["nueva", None]}, index=[9, 9])
        salida = modulo.modulo_bosque.crear_predictor(modelo)(consulta)
        self.assertTrue(salida.index.equals(consulta.index))
        self.assertEqual(imputador.statistics_.tolist(), ["a"])


class TestInterfazMatrices(unittest.TestCase):
    def test_alinea_clases_antes_de_sumar(self):
        raiz = Path(__file__).resolve().parents[1]
        espacio = {}
        notebook = json.loads((raiz / "notebooks/04_interfaz.ipynb").read_text(encoding="utf-8"))
        for celda in notebook["cells"]:
            if "interfaz-definiciones" in celda.get("metadata", {}).get("tags", []):
                exec("".join(celda["source"]), espacio)
        detalle = pd.DataFrame([
            dict(modelo="C4.5", conjunto="prueba", orden_clases=["A", "B"], matriz_confusion=[[1, 2], [3, 4]]),
            dict(modelo="C4.5", conjunto="prueba", orden_clases=["B", "A"], matriz_confusion=[[4, 3], [2, 1]])])
        matriz, clases = espacio["matrices_confusion"](detalle)["C4.5"]
        self.assertEqual(clases, ["A", "B"])
        np.testing.assert_array_equal(matriz, [[2, 4], [6, 8]])


if __name__ == "__main__":
    unittest.main()
