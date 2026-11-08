"""Verifica cobertura y coherencia de los resultados que alimentan el informe."""
from pathlib import Path
import json
import unittest


RAIZ = Path(__file__).resolve().parents[1]


class TestResultadosInforme(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Rechazar NaN no estándar permite transportar el archivo a otros lectores JSON.
        def rechazar_constante(valor):
            raise ValueError(f"Constante no válida en JSON: {valor}")
        cls.resultados = json.loads((RAIZ / "paper/resultados.json").read_text(encoding="utf-8"),
                                    parse_constant=rechazar_constante)

    def test_cobertura_de_tipos_y_evaluacion_final(self):
        finales = self.resultados["finales"]
        self.assertEqual(len(finales), 12)
        self.assertEqual({fila["dataset"] for fila in finales}, {"lenses", "iris", "heart"})
        for fila in finales:
            if fila["dataset"] == "heart" and fila["conjunto"] == "prueba":
                self.assertEqual(fila["cantidad_casos"], 61)
            self.assertEqual(sum(map(sum, fila["matriz_confusion"])), fila["cantidad_casos"])

    def test_faltantes_y_barridos_consolidados(self):
        faltantes = self.resultados["faltantes"]
        barridos = self.resultados["barridos"]
        self.assertEqual(len(faltantes), 48)
        self.assertEqual({fila["fraccion_nulos_objetivo"] for fila in faltantes}, {0, 0.05, 0.15, 0.3})
        self.assertEqual(len(barridos), 162)
        self.assertTrue(all(fila["repeticiones"] == 3 for fila in faltantes + barridos))
        parametros = {fila["parametro_variado"] for fila in barridos}
        self.assertEqual(parametros, {"profundidad_maxima", "minimo_muestras_division", "umbral_ganancia",
            "minima_razon_ganancia", "filtro_ganancia", "numero_arboles", "maximo_atributos",
            "criterio", "usar_bootstrap"})

    def test_las_cifras_de_la_tabla_corresponden_a_los_resultados(self):
        tabla = (RAIZ / "paper/tablas_resultados.tex").read_text(encoding="utf-8")
        for fila in self.resultados["finales"]:
            if fila["conjunto"] == "prueba":
                nombre = {"heart": "Cleveland", "iris": "Iris", "lenses": "Lenses"}[fila["dataset"]]
                fragmento = f"{nombre} & {fila['modelo']} & {fila['accuracy']:.3f} & {fila['f1_macro']:.3f}"
                self.assertIn(fragmento, tabla)


if __name__ == "__main__":
    unittest.main()
