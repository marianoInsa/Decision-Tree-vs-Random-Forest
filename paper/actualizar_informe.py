"""Genera las tablas y observaciones LaTeX a partir de resultados.json."""
import json
from pathlib import Path


MODELOS = ("C4.5", "Random Forest")


def decimal(valor, cifras=3):
    return f"{valor:.{cifras}f}".replace(".", ",")


def generar_tablas(finales, faltantes):
    cierre = r"\bottomrule\end{tabular}\end{table}}"
    lineas = [
        r"\newcommand{\TablaFinal}{%",
        r"\begin{table}[!hbp]\centering\setlength{\tabcolsep}{9pt}",
        r"\caption{Evaluación final con selección interna por validación.}",
        r"\label{tab:finales}\small\begin{tabular}{lrrrr}\toprule",
        r"Modelo & Acierto & F1 macro & Entr. (ms) & Pred. (ms)\\\midrule",
    ]
    for modelo in MODELOS:
        fila = finales[modelo]
        lineas.append(" & ".join((
            modelo, decimal(fila["accuracy"]), decimal(fila["f1_macro"]),
            decimal(fila["tiempo_entrenamiento_ms"], 1), decimal(fila["tiempo_inferencia_ms"], 1),
        )) + r"\\")
    lineas.extend((
        cierre,
        r"\newcommand{\TablaClases}{%",
        r"\begin{table}[!hbp]\centering\setlength{\tabcolsep}{9pt}",
        r"\caption{Evaluación final por clase en el conjunto de prueba.}",
        r"\label{tab:clases}\small\begin{tabular}{llrrrr}\toprule",
        r"Modelo & Clase & Casos & Precisión & Exhaustividad & F1\\\midrule",
    ))
    for modelo in MODELOS:
        fila = finales[modelo]
        matriz = fila["matriz_confusion"]
        for i, clase in enumerate(fila["orden_clases"]):
            aciertos = matriz[i][i]
            reales = sum(matriz[i])
            predichos = sum(renglon[i] for renglon in matriz)
            precision = aciertos / predichos if predichos else 0
            exhaustividad = aciertos / reales if reales else 0
            f1 = 2 * aciertos / (reales + predichos) if reales + predichos else 0
            lineas.append(" & ".join((
                modelo, clase, str(reales), decimal(precision), decimal(exhaustividad), decimal(f1),
            )) + r"\\")
    lineas.extend((
        cierre,
        r"\newcommand{\TablaFaltantes}{%",
        r"\begin{table}[tb]\centering\setlength{\tabcolsep}{9pt}",
        r"\caption{Efecto de los faltantes adicionales. Media de tres repeticiones.}",
        r"\label{tab:nulos}\small\begin{tabular}{lrrrr}\toprule",
        r"Modelo & \multicolumn{2}{c}{Acierto} & \multicolumn{2}{c}{F1 macro}\\",
        r" & 0\% & 30\% & 0\% & 30\%\\\midrule",
    ))
    for modelo in MODELOS:
        filas = faltantes[modelo]
        cifras = [decimal(filas[nivel][metrica]) for metrica in ("accuracy_media", "f1_macro_media")
                  for nivel in (0, 0.3)]
        lineas.append(" & ".join((modelo, *cifras)) + r"\\")
    lineas.append(cierre)
    return "\n".join(lineas) + "\n"


def generar_observaciones(finales, faltantes, barridos):
    arbol = finales["C4.5"]
    bosque = finales["Random Forest"]
    profundidad = arbol["parametros"]["profundidad_maxima"]
    limite = "sin límite de profundidad" if profundidad is None else f"con profundidad máxima {profundidad}"
    cantidad = bosque["parametros"]["numero_arboles"]
    indice_arbol = arbol["orden_clases"].index("presencia")
    indice_bosque = bosque["orden_clases"].index("presencia")
    presentes = sum(arbol["matriz_confusion"][indice_arbol])
    aciertos_arbol = arbol["matriz_confusion"][indice_arbol][indice_arbol]
    aciertos_bosque = bosque["matriz_confusion"][indice_bosque][indice_bosque]
    c45_faltantes = faltantes["C4.5"]
    bosque_faltantes = faltantes["Random Forest"]
    profundidad_libre = barridos["C4.5", "profundidad_maxima", "null"]
    sin_filtro = barridos["C4.5", "filtro_ganancia", "false"]["prueba"]
    con_filtro = barridos["C4.5", "filtro_ganancia", "true"]["prueba"]
    acierto_30, acierto_100, acierto_200 = (
        decimal(barridos["Random Forest", "numero_arboles", valor]["prueba"]["accuracy_media"])
        for valor in ("30", "100", "200")
    )
    observaciones = {
        "Final": f"La validación seleccionó C4.5 {limite} y Random Forest con {cantidad} árboles.",
        "Clases": (f"De los {presentes} casos de presencia, C4.5 identificó {aciertos_arbol} "
                   f"y Random Forest identificó {aciertos_bosque}."),
        "Faltantes": (
            f"Entre 0 y 30\\% de ocultación nominal, el F1 macro medio de C4.5 pasó de "
            f"{decimal(c45_faltantes[0]['f1_macro_media'])} a {decimal(c45_faltantes[0.3]['f1_macro_media'])}. "
            f"En Random Forest pasó de {decimal(bosque_faltantes[0]['f1_macro_media'])} "
            f"a {decimal(bosque_faltantes[0.3]['f1_macro_media'])}."
        ),
        "Barridos": (
            f"Al retirar el límite de profundidad en C4.5, la tasa de acierto media fue "
            f"{decimal(profundidad_libre['entrenamiento']['accuracy_media'])} en entrenamiento y "
            f"{decimal(profundidad_libre['prueba']['accuracy_media'])} en prueba. "
            "La brecha es compatible con un ajuste a particularidades de las muestras de entrenamiento. "
            f"En Random Forest, las medias de prueba con 30, 100 y 200 árboles fueron "
            f"{acierto_30}, {acierto_100} y {acierto_200}, respectivamente. "
            "El valor intermedio obtuvo el promedio más alto entre los tres."
        ),
        "Filtro": (
            f"La tasa de acierto media en prueba fue {decimal(sin_filtro['accuracy_media'])} sin el filtro y "
            f"{decimal(con_filtro['accuracy_media'])} con el filtro. Los desvíos respectivos fueron "
            f"{decimal(sin_filtro['accuracy_desvio'])} y {decimal(con_filtro['accuracy_desvio'])}."
        ),
    }
    return "\n".join(r"\newcommand{\Observacion" + nombre + "}{" + texto + "}"
                     for nombre, texto in observaciones.items()) + "\n"


def actualizar_informe(carpeta=Path(__file__).resolve().parent):
    resultados = json.loads((carpeta / "resultados.json").read_text(encoding="utf-8"))
    finales = {fila["modelo"]: fila for fila in resultados["finales"]
               if fila["dataset"] == "heart" and fila["conjunto"] == "prueba"}
    faltantes = {
        modelo: {fila["fraccion_nulos_objetivo"]: fila for fila in resultados["faltantes"]
                 if fila["dataset"] == "heart" and fila["conjunto"] == "prueba" and fila["modelo"] == modelo}
        for modelo in MODELOS
    }
    barridos = {}
    for fila in resultados["barridos"]:
        if fila["dataset"] == "heart":
            clave = (fila["modelo"], fila["parametro_variado"], fila["valor_parametro"])
            barridos.setdefault(clave, {})[fila["conjunto"]] = fila

    tablas = generar_tablas(finales, faltantes)
    observaciones = generar_observaciones(finales, faltantes, barridos)
    (carpeta / "tablas_resultados.tex").write_text(tablas, encoding="utf-8")
    (carpeta / "observaciones_resultados.tex").write_text(observaciones, encoding="utf-8")


if __name__ == "__main__":
    actualizar_informe()
