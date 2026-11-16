# HU-19: ficha breve para el coloquio

## Qué se comparó

Se comparó un árbol de decisión propio basado en C4.5 con Random Forest de scikit-learn en tres conjuntos de clasificación. Contact Lenses tiene 24 casos y atributos categóricos; Iris tiene 150 casos y atributos numéricos; Cleveland tiene 303 casos y atributos mixtos. Cleveland conserva seis valores ausentes naturales. Los dos modelos reciben los mismos casos de entrenamiento y prueba en cada condición; Random Forest aprende la mediana numérica, la moda categórica y la codificación de categorías solo con entrenamiento.

## Protocolo que respalda el resultado final

Se reserva el 20 % de cada conjunto para prueba mediante una partición estratificada de semilla 42. Del 80 % restante se separa el 25 % para elegir la configuración de cada modelo según F1 macro, que da igual peso a cada clase. Ante un empate se conserva la primera configuración de la grilla. La configuración elegida se reentrena con todo el 80 % inicial y se evalúa en la prueba reservada. Las pruebas adicionales con información incompleta y con distintos parámetros, explicadas a continuación, describen el comportamiento de los modelos y no intervienen en esa elección.

| Conjunto | Casos de prueba | C4.5: acierto / F1 macro | Random Forest: acierto / F1 macro |
| --- | ---: | ---: | ---: |
| Contact Lenses | 5 | 1,000 / 1,000 | 1,000 / 1,000 |
| Iris | 30 | 0,933 / 0,933 | 0,933 / 0,933 |
| Cleveland | 61 | 0,721 / 0,719 | 0,836 / 0,836 |

La tasa de acierto es la proporción de casos clasificados correctamente. F1 macro promedia el equilibrio entre precisión y exhaustividad de cada clase. Los valores de la tabla proceden de `resultados_finales` en prueba y se presentan con tres decimales. La selección final eligió profundidad 2 para C4.5 en Lenses y Cleveland, profundidad sin límite en Iris y 30 árboles para Random Forest en los tres conjuntos.

## Pruebas con información incompleta

Después de la evaluación final, se estudia cómo cambia la capacidad de clasificar cuando falta parte de la información disponible. Para simular esa situación, se ocultan al azar valores de los atributos, tanto en entrenamiento como en prueba. Por ejemplo, se puede ocultar el colesterol de un paciente, conservando su fila y el diagnóstico que se utiliza como respuesta correcta. Los dos modelos reciben las mismas tablas con los mismos valores ocultos. Cada modelo vuelve a entrenarse con la información disponible en esa condición y se evalúa en los casos de prueba correspondientes.

Se prueban cuatro niveles: 0 %, 5 %, 15 % y 30 %. El porcentaje indica la probabilidad de ocultar cada valor de los atributos; por eso, la proporción efectivamente ocultada puede variar. El nivel de 0 % conserva los datos originales, sin agregar ausencias. En Cleveland, esto incluye los seis valores que ya faltaban en el conjunto original.

Cada nivel se repite tres veces para observar el efecto de distintas elecciones aleatorias de valores ocultos, y se calcula el promedio de las métricas. Se utilizan las semillas 42, 43 y 44 para poder reproducir esas elecciones. En el nivel de 0 %, las tres repeticiones conservan la misma tabla, porque no se oculta ningún valor adicional. Los casos asignados a entrenamiento y prueba permanecen fijos durante estas pruebas. También se mantiene una configuración fija de cada modelo, distinta de la seleccionada para la evaluación final; por ello, los resultados de 0 % pueden diferir de los de la tabla anterior.

Al pasar de 0 % a 30 %, el F1 macro medio de C4.5 pasó de 1,000 a 0,230 en Lenses; de 0,933 a 0,803 en Iris; y de 0,703 a 0,790 en Cleveland. En Random Forest pasó de 0,619 a 0,352; de 0,900 a 0,878; y de 0,885 a 0,830, respectivamente. La mejora observada para C4.5 en Cleveland corresponde a estos casos y a estas elecciones de valores ocultos. Ese resultado no permite afirmar que quitar información mejore el modelo en general.

## Pruebas con distintos parámetros

Además de ocultar información, se estudia qué ocurre al cambiar un parámetro del modelo por vez, procedimiento denominado barrido de parámetros. Estas pruebas usan tres repartos distintos de casos entre entrenamiento y prueba, que mantienen aproximadamente las proporciones de las clases, con semillas 42, 43 y 44. En Cleveland, el filtro experimental de C4.5 obtuvo una tasa de acierto media de 0,754 sin activarlo y 0,738 al activarlo; ambos desvíos muestrales fueron 0,043. Las curvas de profundidad y cantidad de árboles muestran medias y desvíos de entrenamiento y prueba. Sus máximos no constituyen una nueva selección de modelo.

## Límites de la interpretación

Los tres problemas son pequeños y distintos; una diferencia entre ellos también puede depender de sus clases, tamaños y atributos. En Lenses, cada acierto en los cinco casos de prueba cambia la tasa en veinte puntos porcentuales. Las pruebas con información incompleta mantienen un único reparto de casos entre entrenamiento y prueba; las pruebas con distintos parámetros usan solo tres repartos, por lo que ambas ofrecen una evaluación limitada. Los tiempos dependen del entorno y pueden cambiar entre ejecuciones; las predicciones y métricas se reprodujeron en la comprobación de HU-19. La comparación describe estas condiciones y no establece una superioridad general de uno de los algoritmos.
