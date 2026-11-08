# Revisión técnica del informe final — HU-18

Fecha: 8 de noviembre de 2026. Versión revisada: commit `ae8da4b1972900f81d79b613df83a88026c79a90` de `main`.

Se revisó el informe LNCS mediante la lectura de sus diez páginas y el contraste de sus explicaciones con los algoritmos, las pruebas y los resultados conservados en el repositorio. El criterio de revisión cruzada de HU-18 queda realizado y documentado. Se detectaron tres correcciones necesarias para precisar la descripción del algoritmo y cuatro ampliaciones recomendadas. No se identificaron errores en las cifras contrastadas ni fallos en las 86 pruebas ejecutadas. Las observaciones siguientes corresponden a la documentación; esta revisión no modifica el informe ni acredita que las correcciones ya estén aplicadas.

## 1. Alcance y fuentes

HU-18 solicita documentar la búsqueda de umbrales, el soporte de tipos mixtos, el reparto fraccionario y los casos borde en el notebook de C4.5, además de verificar técnicamente las explicaciones sobre variables continuas y nulos del informe final. Se tomó la descripción de la tarjeta del archivo de Trello suministrado como especificación de alcance, no como una instrucción para ejecutar acciones externas. Esta revisión se concentra en ese último criterio y comprueba también la coherencia de los resultados y del tratamiento de datos descritos en el informe.

Se consultaron estos archivos:

- [Fuente LaTeX del informe](../paper/Paper_Equipo-QG_TP_IA.tex), [PDF revisado](../paper/Paper_Equipo-QG_TP_IA.pdf), [tablas](../paper/tablas_resultados.tex), [observaciones generadas](../paper/observaciones_resultados.tex) y [resultados exportados](../paper/resultados.json).
- [C4.5](../notebooks/01_decision_tree_c45.ipynb), especialmente las celdas 4, 6 y 10: `evaluar_particion`, `buscar_mejor_umbral`, `ArbolC45._construir` y `ArbolC45._distribucion_fila`. Los números de celda cuentan desde uno e incluyen celdas explicativas.
- [Random Forest](../notebooks/02_random_forest.ipynb), [experimentos](../notebooks/03_experimentos.ipynb) y [preparación compartida de datos](../datos_tp.py).
- Las pruebas de C4.5, experimentos, integración y resultados indicadas en la sección 5.
- Liu, *Web Data Mining*, edición de 2011, disponible en `docs/libros/`: páginas impresas 72–76 y 78–79, correspondientes a las páginas 93–97 y 99–100 del PDF. Se contrastaron entropía, tasa de ganancia, cortes numéricos y reparto de faltantes. Las decisiones particulares del código se verificaron directamente; no se atribuyen todas ellas a esa referencia.

Las ubicaciones del informe que siguen corresponden a la versión identificada arriba. Las líneas son las del archivo `.tex`, no las del PDF.

## 2. Correcciones necesarias

### C1. Hacer explícita la selección global de la división

**Ubicación:** sección 3.1, Algoritmo 1, páginas 4–5; líneas 114–138 del LaTeX, especialmente 124–132.

**Observación:** el pseudocódigo conserva una división por atributo, pero no muestra el paso que compara las candidatas de todos los atributos y selecciona una única división para el nodo. Después utiliza un atributo y sus ramas sin haberlos asignado expresamente. Además, la formulación actual describe únicamente la selección por tasa de ganancia, mientras que la sección experimental incorpora una variante que cambia la búsqueda numérica y filtra candidatas. El código sí contiene ambas operaciones.

**Corrección propuesta:** declarar que el algoritmo resume la configuración por defecto y remitir a la variante, o incorporar una entrada `filtro_ganancia` y mostrar las dos alternativas. En ambos casos, agregar la selección global antes de comprobar los mínimos de ganancia y tasa. Una secuencia fiel al código es:

1. Recibir los casos, sus clases y pesos, los atributos disponibles, la profundidad y los controles de crecimiento. En la llamada inicial, cada caso tiene peso uno.
2. Resumir la masa por clase y aplicar las paradas estructurales.
3. Obtener una candidata por atributo. Para un atributo numérico, conservar el corte de mayor tasa de ganancia ajustada en la configuración por defecto; con el filtro activado, conservar el de mayor ganancia ajustada. Para un atributo categórico, evaluar su partición por valores conocidos.
4. Descartar atributos sin partición válida o con información de partición nula.
5. Si está activado el filtro, conservar las candidatas cuya ganancia ajustada sea al menos la media de las candidatas válidas. El código admite una tolerancia de `1e-12` en esta comparación.
6. Elegir entre las candidatas restantes la de mayor tasa de ganancia ajustada. Ante igualdad exacta, conservar la primera según el orden de atributos.
7. Si no existe candidata o no se alcanzan los controles de crecimiento, devolver una hoja. Los controles de ganancia y tasa se verifican después de elegir la candidata; no se busca otra candidata como reemplazo si esta no los supera.
8. Crear el nodo y construir sus hijos. Cada caso conocido conserva su peso en una sola rama; cada caso faltante aporta a cada rama su peso recibido multiplicado por la proporción de esa rama. Retirar el atributo elegido solo si es categórico y aumentar la profundidad en uno.

**Criterio de resolución:** el pseudocódigo debe permitir identificar sin consultar el código qué división se usa, cómo se transmiten los pesos y qué alcance tiene la variante opcional. Las reglas de desempate del párrafo posterior deben conservarse.

### C2. Separar las masas conocidas de las masas posteriores al reparto

**Ubicación:** sección 2.1, página 3; líneas 69–98 del LaTeX, especialmente las ecuaciones de ganancia ajustada y proporción de rama.

**Observación:** las fórmulas de ganancia ajustada y proporción de rama coinciden con la implementación. Sin embargo, se reutilizan los símbolos de las ramas sin precisar que, al evaluar una candidata, contienen exclusivamente casos con valor conocido para el atributo actual. Después del reparto, los hijos contienen además aportes de casos faltantes. Confundir esos dos momentos cambia el significado de los denominadores. La explicación de la información de partición con faltantes es correcta en prosa, pero conviene escribir la expresión completa para evitar aplicar directamente la fórmula sin faltantes.

**Texto y fórmulas propuestos:**

> Sea $D$ el conjunto de casos que llega al nodo; en la raíz coincide con el bloque de entrenamiento y, en los descendientes, contiene los casos y aportes que llegan a ese nodo. El índice $i$ identifica un caso o fila, $y_i$ es su clase y $w_i$ es el peso con el que participa allí. Inicialmente todos los pesos valen uno. La masa total es $W=\sum_{i\in D}w_i$ y la proporción de la clase $c$ es $p_c=\sum_{i\in D:y_i=c}w_i/W$. Un caso repartido conserva su identidad y su clase; no pasa a representar varios casos completos.

Para evaluar un atributo $A$, definir $K$ como sus casos conocidos en ese nodo, $U$ como sus casos faltantes y $K_j$ como los conocidos que corresponden a la rama $j$, **antes de repartir los faltantes**:

$$
W_K=\sum_{i\in K}w_i,\qquad W_U=\sum_{i\in U}w_i,\qquad
W_j=\sum_{i\in K_j}w_i,\qquad W=W_K+W_U,\qquad \sum_j W_j=W_K.
$$

Con $W_K>0$ y al menos dos ramas conocidas de masa positiva:

$$
G^*(D,A)=\frac{W_K}{W}\left[H(K)-\sum_j\frac{W_j}{W_K}H(K_j)\right],
$$

$$
S^*(D,A)=-\sum_j\frac{W_j}{W}\log_2\frac{W_j}{W}
-\frac{W_U}{W}\log_2\frac{W_U}{W},\qquad
R^*(D,A)=\frac{G^*(D,A)}{S^*(D,A)},\qquad q_j=\frac{W_j}{W_K}.
$$

> Se adopta $0\log_2 0=0$. El término de masa faltante interviene en el cálculo de $S^*$, pero no crea una rama adicional. Si no hay dos ramas conocidas de masa positiva, no se propone esa división; la construcción tampoco selecciona una candidata con $S^*=0$. Al construir el hijo $j$, cada caso de $U$ aporta $w_iq_j$. Por ello, la masa del hijo pasa a ser $W_j+q_jW_U$, mientras que la proporción almacenada continúa siendo la calculada con los conocidos. Como $\sum_jq_j=1$, el reparto conserva la masa total y la masa de cada clase.

**Criterio de resolución:** usar símbolos distintos o aclaraciones inequívocas para las ramas conocidas de evaluación y los hijos posteriores al reparto. Explicitar qué suma cada denominador y evitar una división por cero. La corrección completa la notación; no implica reemplazar un cálculo incorrecto del programa.

### C3. Precisar qué valores generan los cortes y el alcance del criterio

**Ubicación:** sección 2.1, página 3, línea 90; sección experimental, página 9, línea 212; Algoritmo 1.

**Observación:** se mencionan $r-1$ cortes entre valores consecutivos sin definir expresamente $r$ como cantidad de valores conocidos distintos con peso positivo. Si se interpreta como cantidad de filas, los repetidos producirían cortes inexistentes en el código. Además, la frase que conserva el corte de mayor tasa de ganancia requiere acotarse a la configuración por defecto: la variante experimental selecciona el corte numérico por ganancia ajustada antes de filtrar atributos.

**Texto propuesto para reemplazar el inicio del párrafo numérico:**

> En cada nodo se obtienen los valores numéricos conocidos distintos con peso positivo y se ordenan como $v_1<\cdots<v_r$. Si $r<2$, el atributo no ofrece una división. En otro caso se evalúan los $r-1$ umbrales $t=v_1,\ldots,v_{r-1}$, con ramas $A\le t$ y $A>t$. Los valores repetidos permanecen juntos; los faltantes no generan umbrales y los valores decimales no se redondean. Se usa el valor observado inferior de cada separación, sin calcular puntos medios. En la configuración por defecto se conserva el corte de mayor tasa de ganancia ajustada. La variante con filtro, descrita en la sección experimental, elige primero el corte de mayor ganancia ajustada y luego filtra las candidatas por atributo antes de comparar sus tasas. Cada nodo elige un único corte para el atributo seleccionado; el mismo atributo numérico puede volver a utilizarse en sus descendientes.

**Criterio de resolución:** la descripción debe predecir correctamente el comportamiento con valores repetidos, un atributo constante, un caso exactamente igual al umbral y la variante con filtro activado. Conservar la referencia a Liu para los cortes locales y distinguir la variante experimental del trabajo.

## 3. Ampliaciones recomendadas

Estas recomendaciones aumentan la claridad del informe. No se presentan como requisitos nuevos de la tarjeta HU-18 ni como fallos del algoritmo.

### R1. Escribir la regla de inferencia con faltantes y sus excepciones

**Ubicación:** sección 2.1, línea 92, y sección 3.1, línea 138; páginas 3 y 5.

El informe ya indica que se mezclan probabilidades. Se recomienda agregar la expresión $P(c\mid x,n)=\sum_jq_jP(c\mid x,n_j)$ cuando falta el atributo consultado en el nodo $n$. En una hoja, la probabilidad es la masa de la clase dividida por la masa de la hoja. La clase se elige después de combinar las distribuciones completas; no se elige primero la rama con mayor proporción ni se votan las clases ganadoras de las ramas. Los faltantes en columnas que el nodo no consulta no cambian esa decisión.

Conviene conservar la distinción ya correcta entre un valor faltante y una categoría nueva: la categoría no vista utiliza la distribución local del nodo, mientras que el faltante recorre sus hijos con las proporciones aprendidas. Una breve aclaración sobre pesos cero, columnas constantes o completamente faltantes y rechazo de etiquetas faltantes completaría la explicación de casos borde sin reproducir toda la documentación del notebook.

### R2. Acotar las afirmaciones generales sobre Random Forest

**Ubicación:** sección 2.2, líneas 105–107; páginas 3–4.

La descripción general del muestreo con reemplazo y de árboles sin poda debe identificarse como la configuración habitual de referencia. Los experimentos también incluyen `usar_bootstrap=False` y límites de profundidad. Se recomienda indicar que esas variantes modifican las condiciones de construcción del bosque. La ausencia de pospoda no significa que todos los árboles tengan profundidad ilimitada.

También se recomienda separar la votación mayoritaria de clases del promedio de probabilidades: pueden producir respuestas distintas cuando las hojas contienen mezclas de clases. El párrafo siguiente ya describe correctamente el promedio utilizado por scikit-learn. Se puede suprimir la alternativa de votación del primer párrafo o presentarla expresamente como otra regla de agregación.

Para los datos mixtos puede añadirse que los códigos numéricos que representan categorías de Cleveland se convierten en columnas categóricas antes de entrenar. El tipo de almacenamiento y la interpretación del atributo no son equivalentes. En Random Forest, el muestreo de atributos se aplica a las columnas resultantes de la codificación en indicadores.

### R3. Detallar qué se repite en la batería de faltantes

**Ubicación:** sección 5.1, línea 184, y sección 5.2, línea 198; páginas 6 y 8.

Se recomienda explicitar que en la batería de faltantes cambia la semilla de la máscara y se conserva la partición de semilla 42, mientras que en los barridos cambian las semillas de partición. Los porcentajes nominales son probabilidades de ocultación artificial por celda; la fracción observada puede diferir y conserva los faltantes naturales. Las máscaras se construyen sin consultar las clases. Las tres repeticiones del escenario de 0 % no equivalen a tres particiones independientes. La distinción actual entre desvío muestral e intervalo de confianza debe mantenerse.

### R4. Distinguir el costo de ordenar del costo de evaluar todos los cortes

**Ubicación:** sección 2.1, línea 90; página 3.

La afirmación de que ordenar cuesta $O(n\log n)$ es correcta. No debe interpretarse como el costo total de la búsqueda actual: se vuelven a recorrer los casos al evaluar cada uno de los $r-1$ cortes, por lo que el costo por atributo y nodo es $O(n\log n+rn)$ y puede ser cuadrático si casi todos los valores son distintos. El reparto fraccionario también puede aumentar el trabajo en los descendientes. Esta precisión solo es necesaria si el informe pretende informar la complejidad total de la implementación; no exige optimizarla para resolver HU-18.

## 4. Aspectos contrastados sin correcciones detectadas

| Aspecto | Resultado de la revisión |
| --- | --- |
| Cortes numéricos locales | El código usa valores observados, ramas `<=` y `>`, un corte elegido por nodo y reutilización de atributos numéricos. No realiza una discretización global ni redondea decimales. |
| Ganancia con faltantes | Se evalúan los conocidos, se multiplica por $W_K/W$ y se incluye la masa faltante en la información de partición. Coincide con la explicación del informe. |
| Reparto e inferencia | Los pesos recibidos se multiplican por las proporciones; las clases se conservan y la inferencia mezcla distribuciones. No se imputa el valor del atributo en C4.5. |
| Prepoda y desempates | Se distingue prepoda de pospoda; los umbrales de ganancia y tasa son controles separados, la igualdad permite dividir y los empates mantienen el orden documentado. |
| Preparación de Random Forest | Las estadísticas de mediana y moda y la codificación se aprenden con entrenamiento. La política descrita corresponde a la empleada en estos experimentos, aunque el constructor admite otras políticas. |
| Datos | Lenses: 24 casos y cuatro atributos; Iris: 150 y cuatro; Cleveland: 303 y trece. En Cleveland se conservan 139 casos de presencia, 164 de ausencia y seis celdas faltantes. Las huellas de los dos archivos de origen registrados coinciden con las exportadas. |
| Protocolo | La elección interna usa validación; la evaluación final reentrena con el bloque de entrenamiento original. Los barridos se presentan como descriptivos y no seleccionan la configuración final. |
| Escenario de 0 % | El informe aclara que significa cero ausencias artificiales adicionales y conserva las naturales de Cleveland. |
| Interpretación de resultados | Se advierte la sensibilidad de Lenses a sus cinco casos de prueba y no se atribuyen causalmente las diferencias entre problemas al tipo de atributo. No se afirma una superioridad general. |

Se contrastaron las seis filas de la tabla final con `resultados.json`, incluidos sus tiempos redondeados. Los pares de tasa de acierto y F1 macro publicados son:

| Datos | C4.5: acierto / F1 macro | Random Forest: acierto / F1 macro |
| --- | --- | --- |
| Lenses | 1,000 / 1,000 | 1,000 / 1,000 |
| Iris | 0,933 / 0,933 | 0,933 / 0,933 |
| Cleveland | 0,721 / 0,719 | 0,836 / 0,836 |

También se cotejaron los extremos de la tabla de faltantes y las observaciones generadas con los agregados exportados. Por ejemplo, el F1 macro medio de C4.5 en Cleveland pasa de 0,703 a 0,790 entre 0 % y 30 % adicionales: la mejora está registrada y el texto no presupone un deterioro monótono. Las cifras del filtro experimental de Cleveland también coinciden: acierto medio 0,754 sin filtro y 0,738 con filtro, ambos con desvío redondeado 0,043. Esta comprobación verifica coherencia entre archivos; no constituye una nueva ejecución de toda la batería ni una repetición de las mediciones de tiempo.

## 5. Verificación ejecutada y límites

Se ejecutaron las pruebas locales con el intérprete de `.venv` mediante la siguiente selección de suites:

```python
import unittest

suite = unittest.TestSuite()
for patron in (
    "test_c45*.py",
    "test_experimentos.py",
    "test_integracion_pr.py",
    "test_resultados_informe.py",
):
    suite.addTests(unittest.defaultTestLoader.discover("tests", pattern=patron))
resultado = unittest.TextTestRunner(verbosity=1).run(suite)
assert resultado.wasSuccessful()
```

Resultado de la ejecución: **86 pruebas, 1,450 segundos, sin fallos ni errores**. La selección cubre búsqueda numérica, pesos y nulos, crecimiento e inferencia, protocolo experimental, integración de las variantes y coherencia básica de las exportaciones. No se ejecutaron en esta revisión las suites de interfaz ni de compilación del informe. No se volvió a generar el PDF ni a entrenar toda la batería experimental. El PDF existente tiene diez páginas; esta lectura no acredita una comprobación editorial o visual completa de la maquetación.

Las huellas SHA-256 identifican exactamente los archivos revisados:

| Archivo | SHA-256 |
| --- | --- |
| `paper/Paper_Equipo-QG_TP_IA.tex` | `159d4103525bd717aa1c8602710ea142b8727899bba1cec94f293c94737b41a1` |
| `paper/Paper_Equipo-QG_TP_IA.pdf` | `35a2bb64567f70d6749f1be4f1cdf9a7eb20befff7dfd05e9783f8a7260e6ac9` |
| `paper/resultados.json` | `60b010aa2da804dfb56de887b0d24a4b0e06aee188f1e1a964de245207228697` |

## 6. Estado de la revisión y aplicación posterior

La revisión técnica solicitada por HU-18 está completada con observaciones. C1–C3 precisan la correspondencia entre el texto y la implementación; R1–R4 son ampliaciones recomendadas. No se requiere modificar los algoritmos para aplicar estas observaciones. El estado de la tarjeta en Trello no se modifica mediante este documento.

[PENDIENTE DE CORREGIR EN EL INFORME: C1–C3, OBSERVACIONES DE LA REVISIÓN TÉCNICA DE HU-18. NECESARIAS PARA CONSIDERAR RESUELTAS LAS AMBIGÜEDADES DOCUMENTALES IDENTIFICADAS; NO IMPLICAN QUE LAS PRUEBAS DEL ALGORITMO HAYAN FALLADO.]

[MEJORAS OPCIONALES DEL INFORME: R1–R4. NO CONSTITUYEN CRITERIOS DE ACEPTACIÓN ADICIONALES DE HU-18.]

Después de aplicar C1–C3, se deberá recompilar el informe, verificar que las expresiones y el Algoritmo 1 se leen correctamente y conservar el límite de doce páginas. Una ejecución correcta en Google Colab, la revisión de los datos de portada y la publicación remota son actividades distintas: esta revisión no las acredita ni las exige para dar por realizado el contraste técnico solicitado.
