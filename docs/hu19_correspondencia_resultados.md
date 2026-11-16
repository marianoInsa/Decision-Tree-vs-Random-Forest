# HU-19: correspondencia de resultados del informe

## Ejecución y alcance

Se revisó la versión del informe y de los cuadernos correspondiente al commit `6d4b97a`. La exportación publicada es `paper/resultados.json` (SHA-256 `4aa49cdb353ae2cf0f3ed51f523f8b5a7c81784498627879ad3b764dd2ce1c1f`), junto con `paper/tablas_resultados.tex`, `paper/observaciones_resultados.tex` y las figuras de `paper/figuras/`. El archivo registra Python 3.14.7, NumPy 2.5.4, pandas 3.0.6 y scikit-learn 1.9.1. Las fuentes locales de Lenses y Cleveland coinciden con las huellas incluidas en el JSON; Iris proviene de scikit-learn.

El 15 de noviembre de 2026 se ejecutó `paper/generar_resultados.py` en una copia aislada del commit, primero sin cambios y luego con las correcciones documentales de HU-19 en los cuadernos 02 y 03. Se utilizó Python 3.12.8 y las mismas versiones de NumPy, pandas y scikit-learn. Se ejecutaron en orden todas las celdas de código de `03_experimentos.ipynb`; la segunda exportación tiene SHA-256 `ae0e1afa20006a491263ac8181b8f0b7033493275592e01543f66954f1d27b05`. La comparación de los 12 registros de referencia, los 12 finales, los 48 resúmenes de faltantes y los 162 de barridos no encontró diferencias en campos ajenos a versiones y tiempos. Esta igualdad incluye parámetros, índices, predicciones, métricas y matrices en los registros completos, y las medias y desvíos de los dos resúmenes. La ejecución identifica la repetibilidad numérica en este entorno; los tiempos se midieron de nuevo y pueden variar.

## Origen de cada elemento publicado

| Elemento del informe | Origen del cuaderno y conjunto | Configuración, agregación y presentación |
| --- | --- | --- |
| Tabla `tab:finales`, seis filas | Sección 5, celda de exploración y evaluación final; `resultados_finales`, solo `conjunto == 'prueba'`. | Partición estratificada 80/20, semilla 42. Selección por F1 macro en el 25 % del bloque inicial de entrenamiento; reentrenamiento con el 80 %. Lenses y Cleveland: C4.5 con profundidad 2; Iris: sin límite. Random Forest: 30 árboles en los tres conjuntos, con `imputar_frecuente`. Una corrida por fila; acierto y F1 macro a tres decimales; tiempos a un decimal. |
| Tabla `tab:nulos`, seis filas | Sección 6, `resumen_faltantes`, solo prueba y extremos nominales 0 % y 30 %. | Misma partición de semilla 42; máscaras de semillas 42, 43 y 44. C4.5 con profundidad 6 y umbral de ganancia 0,05; Random Forest con 100 árboles y `imputar_frecuente`. Media de tres máscaras; acierto y F1 macro a tres decimales. El 0 % conserva las seis ausencias naturales de Cleveland. |
| Observación de F1 macro final | `paper/actualizar_informe.py`, desde `finales` en prueba. | Los mismos seis valores de `tab:finales`, a tres decimales. |
| Observación de faltantes | `paper/actualizar_informe.py`, desde `faltantes` en prueba. | F1 macro medio de 0 % y 30 % para cada conjunto y modelo, a tres decimales. |
| Observación del filtro | Sección 8.1, `resumen_barridos` de Cleveland, C4.5, `filtro_ganancia`, prueba. | Particiones estratificadas de semillas 42, 43 y 44; media y desvío muestral de la tasa de acierto, a tres decimales. Filtro desactivado: 0,754 ± 0,043; activado: 0,738 ± 0,043. Este barrido no selecciona la evaluación final. |
| Figura `fig:nulos` | Sección 6, `resumen_faltantes`; `paper/figuras/faltantes_resumen.pdf`. | F1 macro medio de prueba para 0 %, 5 %, 15 % y 30 % nominales; las series se trazan desde el resumen sin redondeo previo. |
| Figura `fig:profundidad` | Sección 8.1, `resumen_barridos`; `paper/figuras/profundidad_c45.pdf`. | Cleveland, C4.5, profundidades 2, 4 y sin límite. Tasa de acierto media y desvío muestral en entrenamiento y prueba entre tres particiones. |
| Figura `fig:arboles` | Sección 8.1, `resumen_barridos`; `paper/figuras/arboles_bosque.pdf`. | Cleveland, Random Forest, 30, 100 y 200 árboles. Tasa de acierto media y desvío muestral en entrenamiento y prueba entre tres particiones. |

`paper/tablas_resultados.tex` se genera desde las tablas del cuaderno; `paper/actualizar_informe.py` genera las observaciones desde el JSON. El PDF publicado tiene diez páginas y muestra los mismos valores de las dos tablas y las tres observaciones. Las tres figuras incluidas en el informe se regeneraron con la misma imagen rasterizada a 110 puntos por pulgada. Las figuras de barridos y faltantes del cuaderno también coincidieron. `comparacion.pdf`, que no se inserta en el informe, cambió en sus paneles de tiempos al repetir la ejecución.

## Valores comprobados y discrepancias

| Conjunto | C4.5: acierto / F1 macro final | Random Forest: acierto / F1 macro final | C4.5: F1 macro 0 % → 30 % | Random Forest: F1 macro 0 % → 30 % |
| --- | ---: | ---: | ---: | ---: |
| Lenses | 1,000 / 1,000 | 1,000 / 1,000 | 1,000 → 0,230 | 0,619 → 0,352 |
| Iris | 0,933 / 0,933 | 0,933 / 0,933 | 0,933 → 0,803 | 0,900 → 0,878 |
| Cleveland | 0,721 / 0,719 | 0,836 / 0,836 | 0,703 → 0,790 | 0,885 → 0,830 |

No se encontraron discrepancias numéricas en las métricas publicadas. La tabla `tab:finales` sí conserva tiempos de la ejecución publicada que difieren de la repetición. Los pares siguientes son entrenamiento / inferencia de prueba, en milisegundos y con el mismo redondeo de la tabla:

| Fila de `tab:finales` | Publicado | Nueva ejecución | Explicación |
| --- | ---: | ---: | --- |
| Lenses, C4.5 | 2,0 / 0,7 | 1,5 / 0,6 | Medición de tiempo variable. |
| Lenses, Random Forest | 30,3 / 6,5 | 23,2 / 3,9 | Medición de tiempo variable. |
| Iris, C4.5 | 150,0 / 1,3 | 89,0 / 0,8 | Medición de tiempo variable. |
| Iris, Random Forest | 35,6 / 5,2 | 26,3 / 3,3 | Medición de tiempo variable. |
| Cleveland, C4.5 | 907,9 / 3,5 | 410,8 / 1,9 | Medición de tiempo variable. |
| Cleveland, Random Forest | 57,5 / 13,6 | 34,0 / 6,8 | Medición de tiempo variable. |

La ejecución publicada usó otra versión de Python. Esa diferencia y la variación normal de las mediciones impiden exigir igualdad de tiempos; no se atribuye el cambio a una sola causa. La diferencia documental estaba en la sección 6 del cuaderno: se llamaba `imputar` a la política que el código configura como `imputar_frecuente`. Se corrigió esa descripción sin cambiar la batería. También se precisó qué resultados completos y qué resúmenes contiene la exportación. No se eliminó ninguna celda; se conservaron los ejemplos, las comprobaciones y las etiquetas de definiciones que usan otros cuadernos.

Para HU-15 no se requiere cambiar las cifras ni el texto de resultados del informe con la evidencia revisada. Para HU-16 se entrega esta correspondencia como control de integración: si se regeneran tablas o figuras, corresponde revisar juntos el JSON, las tablas, las observaciones, las tres figuras y el PDF de cierre. Los tiempos de una nueva ejecución pueden diferir sin que cambien las predicciones ni las métricas. La ficha de apoyo se encuentra en `docs/hu19_ficha_coloquio.md`.

## Comprobaciones de los cambios

Se cargaron las definiciones etiquetadas de ambos cuadernos y pasaron las 16 pruebas de `tests/test_experimentos.py` y las tres de `tests/test_resultados_informe.py`. También se compiló una copia del informe: produjo diez páginas, dentro del límite de doce. La comparación estructural de los cuadernos con el commit inicial confirmó que se conservaron la cantidad de celdas, las etiquetas, el código ejecutable y las salidas guardadas; solo cambió documentación. La ejecución completa con esos cambios confirmó la igualdad de métricas indicada arriba. La comprobación local no reemplaza la aceptación integral en Colab de HU-20 ni la aprobación editorial de HU-16.
