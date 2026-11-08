"""Verificaciones de integración del código real de los tres notebooks."""
import ast
import json
from pathlib import Path
import types
import unittest

import numpy as np
import pandas as pd


RAIZ = Path(__file__).resolve().parents[1]
modulo = types.ModuleType("experimentos_desde_notebook")
for celda in json.loads((RAIZ / "notebooks/03_experimentos.ipynb").read_text(encoding="utf-8"))["cells"]:
    if "experimentos-definiciones" in celda.get("metadata", {}).get("tags", []):
        exec(compile("".join(celda["source"]), "03_experimentos.ipynb", "exec"), modulo.__dict__)


def conjuntos_pequenos():
    caracteristicas = pd.DataFrame({"medida": [1, 2, 3, 4, 1.5, 3.5, 1.2, 3.2],
                                    "grupo": ["a", "a", "b", "b", "a", "b", "a", "b"]},
                                   index=[20, 4, 18, 50, 7, 9, 15, 31])
    objetivos = pd.DataFrame({"objetivo": ["A", "A", "B", "B", "A", "B", "A", "B"]},
                             index=caracteristicas.index)
    return {nombre: (caracteristicas.iloc[posiciones].copy(), objetivos.iloc[posiciones].copy())
            for nombre, posiciones in {"entrenamiento": [0, 1, 2, 3],
                                       "validacion": [4, 5], "prueba": [6, 7]}.items()}


class TestControlesArbol(unittest.TestCase):
    def test_limites_estructurales(self):
        caracteristicas, objetivos = conjuntos_pequenos()["entrenamiento"]
        for parametros, motivo in ((dict(profundidad_maxima=0), "profundidad_maxima"),
                                    (dict(minimo_muestras_division=5), "masa_menor_que_min_samples_split")):
            with self.subTest(parametros=parametros):
                arbol = modulo.modulo_arbol.ArbolC45(**parametros).entrenar(caracteristicas, objetivos)
                self.assertTrue(arbol.raiz_.es_hoja)
                self.assertEqual(arbol.raiz_.motivo_parada, motivo)

    def test_ganancia_y_razon_son_controles_distintos(self):
        caracteristicas = pd.DataFrame({"medida": [1, 2, None]})
        objetivos = pd.DataFrame({"objetivo": ["A", "B", "A"]})
        arbol = modulo.modulo_arbol.ArbolC45(umbral_ganancia=0.5).entrenar(caracteristicas, objetivos)
        self.assertFalse(arbol.raiz_.es_hoja)
        arbol = modulo.modulo_arbol.ArbolC45(minima_razon_ganancia=0.5).entrenar(caracteristicas, objetivos)
        self.assertEqual(arbol.raiz_.motivo_parada, "gain_ratio_menor_que_min_gain_ratio")

    def test_limites_invalidos_y_nombres_compatibles(self):
        for parametros in (dict(profundidad_maxima=-1), dict(minimo_muestras_division=1),
                           dict(minima_razon_ganancia=float("nan")), dict(max_depth=True)):
            with self.subTest(parametros=parametros), self.assertRaises(ValueError):
                modulo.modulo_arbol.ArbolC45(**parametros)
        arbol = modulo.modulo_arbol.ArbolC45(max_depth=3, min_samples_split=4, min_gain_ratio=0.1)
        self.assertEqual(arbol.obtener_parametros(), dict(umbral_ganancia=0.0,
            profundidad_maxima=3, minimo_muestras_division=4, minima_razon_ganancia=0.1))


class TestBosque(unittest.TestCase):
    def test_parametros_y_contrato(self):
        caracteristicas, objetivos = conjuntos_pequenos()["entrenamiento"]
        original = caracteristicas.copy(deep=True)
        bosque = modulo.modulo_bosque.entrenar_bosque(caracteristicas, objetivos,
            numero_arboles=3, maximo_atributos="log2", profundidad_maxima=2,
            criterio="entropy", usar_bootstrap=False, semilla=7)
        estimador = bosque.named_steps["random_forest"]
        self.assertEqual((estimador.n_estimators, estimador.max_features, estimador.max_depth,
                          estimador.criterion, estimador.bootstrap, estimador.random_state),
                         (3, "log2", 2, "entropy", False, 7))
        consulta = caracteristicas.iloc[[3, 0, 3]]
        salida = modulo.modulo_bosque.crear_predictor(bosque)(consulta)
        self.assertEqual(list(salida.columns), ["prediccion"])
        self.assertTrue(salida.index.equals(consulta.index))
        pd.testing.assert_frame_equal(caracteristicas, original)

    def test_imputacion_solo_con_entrenamiento_y_columnas_vacias(self):
        caracteristicas = pd.DataFrame({"medida": [1.0, 2.0, 100.0, np.nan],
            "categoria": ["a", None, "b", "a"], "numero_vacio": [np.nan] * 4,
            "categoria_vacia": [None] * 4})
        objetivos = pd.DataFrame({"objetivo": ["A", "A", "B", "B"]})
        original = caracteristicas.copy(deep=True)
        bosque = modulo.modulo_bosque.entrenar_bosque(caracteristicas, objetivos,
            numero_arboles=3, tratamiento_faltantes="imputar")
        imputador = bosque.named_steps["preprocesador"].named_transformers_["numericas"]
        np.testing.assert_array_equal(imputador.statistics_, [2, 0])
        consulta = pd.DataFrame({"medida": [1000.0, None], "categoria": ["nueva", None],
                                 "numero_vacio": [None, None], "categoria_vacia": [None, None]})
        salida = modulo.modulo_bosque.crear_predictor(bosque)(consulta)
        self.assertEqual(len(salida), 2)
        np.testing.assert_array_equal(imputador.statistics_, [2, 0])
        pd.testing.assert_frame_equal(caracteristicas, original)

    def test_politica_rechazar_y_configuraciones_invalidas(self):
        caracteristicas, objetivos = conjuntos_pequenos()["entrenamiento"]
        with self.assertRaises(ValueError):
            modulo.modulo_bosque.entrenar_bosque(caracteristicas.assign(medida=np.nan), objetivos)
        for parametros in (dict(numero_arboles=0), dict(maximo_atributos=False),
                           dict(profundidad_maxima=0), dict(criterio="otro"),
                           dict(usar_bootstrap="si"), dict(semilla=-1)):
            with self.subTest(parametros=parametros), self.assertRaises(ValueError):
                modulo.modulo_bosque.crear_bosque(caracteristicas, **parametros)


class TestMotor(unittest.TestCase):
    def test_producto_grillas_registros_y_repetibilidad(self):
        conjuntos = {nombre: valor for nombre, valor in conjuntos_pequenos().items() if nombre != "prueba"}
        grillas = {"C4.5": {"profundidad_maxima": [0, None]},
                   "Random Forest": {"numero_arboles": [2, 3]}}
        resultados = modulo.ejecutar_grillas("sintetico", "mixto", conjuntos, grillas)
        repetidos = modulo.ejecutar_grillas("sintetico", "mixto", conjuntos, grillas)
        self.assertEqual(len(resultados), 8)
        self.assertEqual(resultados.id_ejecucion.nunique(), 4)
        for _, fila in resultados.iterrows():
            self.assertEqual(fila.indices, conjuntos[fila.conjunto][0].index.tolist())
            self.assertEqual(np.array(fila.matriz_confusion).sum(), fila.cantidad_casos)
            self.assertEqual(len(fila.predicciones), fila.cantidad_casos)
        columnas = [columna for columna in resultados if not columna.startswith("tiempo_")]
        pd.testing.assert_frame_equal(resultados[columnas], repetidos[columnas])

    def test_seleccion_usa_validacion_y_rechaza_prueba(self):
        conjuntos = conjuntos_pequenos()
        resultados = modulo.ejecutar_grillas("sintetico", "mixto",
            {nombre: valor for nombre, valor in conjuntos.items() if nombre != "prueba"},
            {"C4.5": {"profundidad_maxima": [0, None]}})
        elegidas = modulo.seleccionar_por_validacion(resultados)
        self.assertEqual(elegidas["C4.5"]["profundidad_maxima"], [None])
        resultados.loc[resultados.conjunto.eq("validacion"), "conjunto"] = "prueba"
        with self.assertRaisesRegex(ValueError, "prueba"):
            modulo.seleccionar_por_validacion(resultados)

    def test_errores_no_se_omiten_y_explican_configuracion(self):
        conjuntos = conjuntos_pequenos()
        with self.assertRaisesRegex(ValueError, "Random Forest.*configuración.*numero_arboles.*causa"):
            modulo.ejecutar_grillas("sintetico", "mixto", conjuntos,
                                    {"Random Forest": {"numero_arboles": [0]}})
        with self.assertRaisesRegex(ValueError, "C4.5.*grilla"):
            modulo.ejecutar_grillas("sintetico", "mixto", conjuntos,
                                    {"C4.5": {"profundidad_maxima": []}})
        conjuntos["prueba"] = conjuntos["entrenamiento"]
        with self.assertRaisesRegex(ValueError, "compartidos"):
            modulo.validar_conjuntos(conjuntos)

    def test_ceros_y_orden_de_clases(self):
        objetivos = pd.DataFrame({"objetivo": ["B", "B"]}, index=[8, 3])
        predicciones = pd.DataFrame({"prediccion": ["A", "A"]}, index=[8, 3])
        medidas = modulo.calcular_metricas(objetivos, predicciones, ["B", "A"])
        self.assertEqual(medidas["matriz_confusion"], [[0, 2], [0, 0]])
        self.assertEqual(medidas["f1_macro"], 0)
        self.assertEqual(medidas["precision_weighted"], 0)

    def test_mascara_reproducible_compartida_sin_mutacion(self):
        conjuntos = conjuntos_pequenos()
        primera = modulo.inyectar_faltantes(conjuntos, 0.5, 42)
        segunda = modulo.inyectar_faltantes(conjuntos, 0.5, 42)
        for nombre, (caracteristicas, objetivos) in conjuntos.items():
            pd.testing.assert_frame_equal(primera[nombre][0], segunda[nombre][0])
            pd.testing.assert_frame_equal(primera[nombre][1], objetivos)
            self.assertFalse(caracteristicas.isna().any().any())


class TestBateriaFaltantes(unittest.TestCase):
    def test_cobertura_y_repetibilidad_de_los_cuatro_porcentajes(self):
        conjuntos = conjuntos_pequenos()
        referencia = {nombre: valor for nombre, valor in conjuntos.items() if nombre != "validacion"}
        datos = {"sintetico": {"tipo": "mixto"}}
        particiones = {"sintetico": {"referencia": referencia}}
        grillas = {"C4.5": {"profundidad_maxima": [2]},
                   "Random Forest": {"numero_arboles": [2], "tratamiento_faltantes": ["imputar"]}}
        primera = modulo.ejecutar_bateria_faltantes(datos, particiones, grillas, semillas_nulos=(42, 43))
        segunda = modulo.ejecutar_bateria_faltantes(datos, particiones, grillas, semillas_nulos=(42, 43))
        self.assertEqual(len(primera), 32)
        self.assertEqual(primera.id_ejecucion.nunique(), 16)
        self.assertFalse(primera.duplicated(["id_ejecucion", "conjunto"]).any())
        self.assertEqual(set(primera.fraccion_nulos_objetivo), {0, 0.05, 0.15, 0.30})
        columnas = [columna for columna in primera if not columna.startswith("tiempo_")]
        pd.testing.assert_frame_equal(primera[columnas], segunda[columnas])
        diferencias, resumen = modulo.resumir_bateria_faltantes(primera)
        self.assertTrue(resumen.repeticiones.eq(2).all())
        self.assertTrue(diferencias.loc[diferencias.fraccion_nulos_objetivo.eq(0),
            ["caida_accuracy_pp", "caida_f1_macro_pp"]].eq(0).all().all())
        # Ambos algoritmos evalúan exactamente los mismos casos y proporciones ausentes.
        for _, pareja in primera.groupby(["conjunto", "repeticion", "fraccion_nulos_objetivo"]):
            self.assertEqual(pareja.iloc[0].indices, pareja.iloc[1].indices)
            self.assertEqual(pareja.iloc[0].fraccion_nulos_observada,
                             pareja.iloc[1].fraccion_nulos_observada)
        for caracteristicas, _ in referencia.values():
            self.assertFalse(caracteristicas.isna().any().any())

    def test_resumen_y_caidas_con_valores_conocidos(self):
        filas = []
        for repeticion, (accuracy_base, accuracy_nulos) in enumerate(((0.8, 0.4), (0.6, 0.8)), 1):
            for fraccion, exactitud in ((0, accuracy_base), (0.3, accuracy_nulos)):
                fila = dict(dataset="manual", tipo_datos="mixto", modelo="C4.5", conjunto="prueba",
                    configuracion=1, repeticion=repeticion, semilla_nulos=40 + repeticion,
                    fraccion_nulos_objetivo=fraccion, fraccion_nulos_observada=fraccion,
                    tiempo_entrenamiento_ms=10, tiempo_inferencia_ms=2)
                fila.update({metrica: exactitud for metrica in modulo.METRICAS})
                filas.append(fila)
        diferencias, resumen = modulo.resumir_bateria_faltantes(pd.DataFrame(filas))
        nivel = resumen.loc[resumen.fraccion_nulos_objetivo.eq(0.3)].iloc[0]
        self.assertAlmostEqual(nivel.accuracy_media, 0.6)
        self.assertAlmostEqual(nivel.accuracy_desvio, np.sqrt(0.08))
        self.assertAlmostEqual(nivel.caida_accuracy_pp_media, 10)
        np.testing.assert_allclose(diferencias.loc[diferencias.fraccion_nulos_objetivo.eq(0.3),
                                                   "caida_accuracy_pp"], [40, -20])

    def test_mascaras_anidadas_y_etiquetas_intactas(self):
        conjuntos = conjuntos_pequenos()
        menor = modulo.inyectar_faltantes(conjuntos, 0.05, 42)
        mayor = modulo.inyectar_faltantes(conjuntos, 0.30, 42)
        for nombre, (_, objetivos) in conjuntos.items():
            self.assertTrue((~menor[nombre][0].isna() | mayor[nombre][0].isna()).all().all())
            pd.testing.assert_frame_equal(mayor[nombre][1], objetivos)

    def test_no_acepta_bateria_sin_referencia_o_con_varios_parametros(self):
        grillas = {"C4.5": {}, "Random Forest": {}}
        for opciones in (dict(fracciones=(0.05, 0.30)), dict(fracciones=(0, 0.3, 0.3)),
                         dict(semillas_nulos=()), dict(semillas_nulos=(42, 42))):
            with self.subTest(opciones=opciones), self.assertRaises(ValueError):
                modulo.ejecutar_bateria_faltantes({}, {}, grillas, **opciones)
        with self.assertRaisesRegex(ValueError, "configuración fija"):
            modulo.ejecutar_bateria_faltantes({}, {},
                {"C4.5": {"profundidad_maxima": [1, 2]}, "Random Forest": {}})


class TestOrganizacionNotebooks(unittest.TestCase):
    def test_notebooks_algoritmicos_sin_ejecuciones_de_ejemplo(self):
        for nombre in ("01_decision_tree_c45.ipynb", "02_random_forest.ipynb"):
            notebook = json.loads((RAIZ / "notebooks" / nombre).read_text(encoding="utf-8"))
            for celda in notebook["cells"]:
                if celda["cell_type"] != "code":
                    continue
                for sentencia in ast.parse("".join(celda["source"])).body:
                    self.assertIsInstance(sentencia, (ast.Import, ast.ImportFrom, ast.FunctionDef,
                                                      ast.ClassDef, ast.Assign))
                    if isinstance(sentencia, ast.Assign):
                        self.assertIsInstance(sentencia.value, ast.Name)


if __name__ == "__main__":
    unittest.main()
