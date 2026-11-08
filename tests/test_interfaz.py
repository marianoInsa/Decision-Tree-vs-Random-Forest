"""Pruebas de la interfaz de 04_interfaz.ipynb, sin navegador."""

import json
from pathlib import Path
import unittest

import matplotlib

matplotlib.use("Agg")

RAIZ = Path(__file__).resolve().parents[1]
NOTEBOOK = RAIZ / "notebooks/04_interfaz.ipynb"
espacio = {}
for celda in json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]:
    if "interfaz-definiciones" in celda.get("metadata", {}).get("tags", []):
        exec(compile("".join(celda["source"]), str(NOTEBOOK), "exec"), espacio)

C45 = {"max_depth": 3, "min_samples_split": 2, "threshold": 0.0, "min_gain_ratio": 0.0}
RF = {"n_estimators": 10, "max_features": "sqrt", "criterion": "gini", "bootstrap": True, "max_depth": None}


class TestInterfaz(unittest.TestCase):
    def test_comparar_usa_la_misma_particion_y_resume_ambos_modelos(self):
        detalle = espacio["comparar"]("heart", C45, RF, repeticiones=2)
        self.assertEqual(len(detalle), 8)
        test = detalle[detalle["conjunto"] == "prueba"]
        self.assertEqual(set(test["cantidad_casos"]), {61})
        resumen = espacio["resumir"](detalle)
        self.assertEqual(list(resumen.index), ["C4.5", "Random Forest"])
        self.assertTrue(resumen["Accuracy"].between(0, 1).all())
        for matriz, _ in espacio["matrices_confusion"](detalle).values():
            self.assertEqual(matriz.sum(), 122)

    def test_barrer_devuelve_una_curva_por_conjunto(self):
        curvas = espacio["barrer"]("iris", "C4.5", "max_depth", [1, 3], C45, RF, repeticiones=2)
        self.assertEqual(len(curvas), 4)
        self.assertEqual(set(curvas["valor"]), {1, 3})

    def test_arbol_dot_respeta_la_profundidad(self):
        arbol = espacio["entrenar_arbol"]("heart", {"max_depth": 4})
        nodos, hojas, profundidad = espacio["medir_arbol"](arbol.raiz_)
        self.assertLessEqual(profundidad, 4)
        self.assertGreater(nodos, hojas)
        dot = espacio["arbol_dot"](arbol.raiz_, 1)
        self.assertTrue(dot.startswith("digraph"))
        self.assertIn("sin dibujar", dot)

    def test_botones_ejecutan_y_avisan(self):
        interfaz = espacio["crear_interfaz"]()
        c = interfaz.controles
        c.repeticiones.value = 1
        c.rf_arboles.value = 10
        for clave in ("comparar", "barrer", "arbol"):
            interfaz.botones[clave].click()
            self.assertIn("Listo", interfaz.estado.value, clave)
        c.barrido_valores.value = "a, b"
        interfaz.botones["barrer"].click()
        self.assertIn("No se pudo ejecutar", interfaz.estado.value)

    def test_cambiar_modelo_actualiza_parametros_y_valores(self):
        c = espacio["crear_interfaz"]().controles
        c.barrido_modelo.value = "Random Forest"
        self.assertIn("n_estimators", c.barrido_parametro.options)
        c.barrido_parametro.value = "n_estimators"
        self.assertIn("200", c.barrido_valores.value)

    def test_cargar_configuracion_elegida_por_validacion(self):
        interfaz = espacio["crear_interfaz"]()
        controles = interfaz.controles
        controles.dataset.value = "heart"
        controles.usar_validada.click()
        self.assertIn("validación cargados", interfaz.estado.value)
        resultados = json.loads((RAIZ / "paper/resultados.json").read_text(encoding="utf-8"))
        elegidos = {fila["modelo"]: fila["parametros"] for fila in resultados["finales"]
                    if fila["dataset"] == "heart" and fila["conjunto"] == "prueba"}
        self.assertEqual(controles.rf_arboles.value, elegidos["Random Forest"]["numero_arboles"])
        self.assertEqual(controles.c45_filtro.value, elegidos["C4.5"]["filtro_ganancia"])
        self.assertEqual(controles.c45_limite.value, elegidos["C4.5"]["profundidad_maxima"] is not None)

    def test_rechaza_repeticiones_y_particiones_invalidas(self):
        for parametros in (dict(repeticiones=0), dict(semilla=-1), dict(fraccion_test=0)):
            with self.subTest(parametros=parametros), self.assertRaises(ValueError):
                espacio["comparar"]("iris", C45, RF, **parametros)


if __name__ == "__main__":
    unittest.main()
