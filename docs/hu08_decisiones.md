# HU-08 — Faltantes e inferencia probabilística en C4.5

La implementación está en las celdas de definiciones de
`notebooks/01_decision_tree_c45.ipynb`. Extiende el árbol de HU-02/HU-03; no
imputa datos, no usa scikit-learn para aprender el árbol y no modifica Random Forest.

## Decisiones y recomendaciones adoptadas

En esta tabla, **atributo** es una característica de entrada y **clase** es la
respuesta que queremos predecir. El **peso** indica cuánto aporta cada ejemplo
al cálculo: empieza en 1 y puede repartirse entre varias ramas cuando falta un
valor.

| ¿Qué situación hay que resolver? | ¿Qué hace la implementación? | ¿Por qué se decidió así? |
| --- | --- | --- |
| **¿Qué valores se consideran faltantes en los atributos?** | Reconoce `None`, `NaN`, `pd.NA` y `NaT` como valores faltantes. | Son representaciones habituales de datos ausentes. No se interpretan como una categoría más del atributo. |
| **¿Qué pasa si falta la clase de un ejemplo de entrenamiento?** | Rechaza ese ejemplo como entrada válida para entrenar. | Sin la clase, no se conoce la respuesta que el árbol debería aprender. |
| **¿Cómo se utiliza un ejemplo de entrenamiento si le falta el atributo que decide la división?** | Lo envía a todas las ramas, repartiendo su peso según la proporción de peso de los ejemplos conocidos que va a cada una. | Permite aprovechar el ejemplo sin inventar el valor faltante. La suma de sus aportes conserva su peso original, incluso después de varias divisiones. |
| **¿Cómo se calcula la ganancia de información cuando un atributo tiene faltantes?** | Calcula la ganancia usando los ejemplos con valor conocido y la multiplica por la proporción del peso total que representan esos ejemplos. | Así, la ganancia refleja cuánto respaldo tiene en los datos disponibles. |
| **¿Cómo se incluyen los faltantes en el cálculo de Split Info?** | Incluye el peso de los ejemplos faltantes como un grupo adicional al calcular Split Info, que es el denominador de Gain Ratio. Ese grupo se usa solo en el cálculo; no crea una rama del árbol. | Hace que la evaluación de la división también tenga en cuenta los datos desconocidos. |
| **¿Se puede dividir por un atributo si todos sus valores faltan o si todos los conocidos son iguales?** | No utiliza ese atributo para dividir. Si ningún atributo permite una división válida, crea una hoja con las proporciones de clases del nodo. | Ese atributo no permite separar los ejemplos en grupos distintos con información conocida. |
| **¿Cómo se predice cuando falta el atributo que indica qué rama seguir?** | Consulta todas las ramas y combina sus probabilidades de clase, usando las proporciones calculadas durante el entrenamiento. | Conserva la información de todas las clases. Tomar solamente la clase ganadora de cada rama perdería información y podría cambiar el resultado. |
| **¿Qué pasa si al predecir aparece una categoría que no se vio en ese nodo durante el entrenamiento?** | Usa las proporciones de clases del nodo alcanzado. La clase predicha es la de mayor proporción. | Mantiene el comportamiento acordado para categorías nuevas, que se tratan de forma distinta a los valores faltantes. |
| **¿En qué formato se devuelven las probabilidades de cada clase?** | `predict_proba` devuelve una tabla de pandas: una fila por ejemplo y una columna por clase. Conserva el índice y el orden de las filas de entrada; las columnas siguen el orden de `classes_`. | Permite identificar a qué ejemplo y a qué clase corresponde cada probabilidad. |
| **¿Qué clase se elige si varias tienen la misma probabilidad máxima?** | Elige la que apareció primero en el conjunto completo de entrenamiento. Ese mismo orden se usa en `classes_`. `predict` siempre elige la clase con mayor probabilidad de `predict_proba`, aplicando ese desempate. | Evita que la etiqueta predicha y las probabilidades devueltas utilicen criterios diferentes. |
| **¿Qué división se elige si varias tienen exactamente el mismo Gain Ratio?** | Prioriza el atributo que aparece primero en el orden de columnas. Entre cortes numéricos empatados de un mismo atributo, elige el menor valor de corte. | Hace que el resultado sea reproducible y conserva el criterio del árbol base. |
| **¿Cuándo se detiene el crecimiento del árbol por falta de ganancia?** | Impide una división si su ganancia, ya ajustada por los faltantes, es menor que `threshold`. Si es igual, este criterio no la impide. | La decisión de seguir dividiendo también debe considerar cuánta información conocida respalda la ganancia. |
| **¿Qué pasa con los ejemplos cuyo peso queda muy pequeño después de varios repartos?** | Conserva todos los pesos positivos sin redondearlos. Solo normaliza las probabilidades finales para que sumen 1. | Evita descartar arbitrariamente ejemplos que todavía aportan información. |

Las decisiones sobre **categorías nuevas, formato de las probabilidades y orden de
las clases** fueron confirmadas por el responsable de la tarea. Las demás son
recomendaciones de implementación documentadas para revisión del equipo.

## Cálculo en un nodo

Sean `W` la masa total, `W_K` la masa de casos con valor conocido para el
atributo, `W_U` la masa faltante y `W_j` la masa conocida de la rama `j`.
Cada caso empieza con peso 1; los descendientes pueden recibir pesos fraccionarios.
Para un conjunto con masa `W`, `p_c` se calcula como masa de clase `c` dividida por `W`.

```text
H(D) = -sum_c p_c log2(p_c)
Gain_conocido = H(D_K) - sum_j (W_j / W_K) H(D_j)
Gain_ajustado = (W_K / W) Gain_conocido
SplitInfo = -sum_j (W_j / W) log2(W_j / W)
            - (W_U / W) log2(W_U / W)
GainRatio = Gain_ajustado / SplitInfo
q_j = W_j / W_K
```

Se usa la convención `0 log2(0) = 0`. Una partición requiere al menos dos
ramas conocidas de masa positiva; `W_K = 0` o un atributo conocido constante
no generan candidatos. Los cortes numéricos usan solo valores conocidos del nodo,
con el mismo criterio `A <= v_i` / `A > v_i` del árbol base.

Para cada ejemplo faltante de peso `w`, el hijo `j` recibe `w * q_j`.
La inferencia usa `P(c | fila, nodo) = sum_j q_j P(c | fila, hijo_j)` cuando
el atributo de decisión está faltante. En hojas o categorías nuevas usa las
frecuencias ponderadas del nodo. Son probabilidades empíricas, sin calibración.

Este desarrollo concreta el enfoque probabilístico descrito en la sección 8 de
`docs/guia_algoritmos.md` (Liu, pp. 78–79). Las convenciones de API, desempates,
validación y parada son decisiones del proyecto.

## Ejemplo verificable a mano

Para `x = [1, 2, None]` e `y = [A, B, A]`, el corte es `x <= 1`:

- `Gain_ajustado = 2/3`, `SplitInfo = log2(3)` y `q_izq = q_der = 1/2`.
- La izquierda tiene masa `A = 1.5`; la derecha, `A = 0.5` y `B = 1`.
- Para `x` faltante, `P(A) = 2/3` y `P(B) = 1/3`.
- Para `x = 2`, `P(A) = 1/3` y `P(B) = 2/3`.

Las pruebas contrastan estos valores y otros cálculos manuales, verifican la
conservación de masa por clase en varios niveles y distinguen mezcla de
distribuciones de voto de mayorías.

## Compatibilidad y límites

- `fit(X, y)` y `predict(X)` mantienen sus firmas. El contrato compartido con RF
  sigue siendo `predict`; HU-08 agrega capacidades propias de C4.5.
- El cambio deliberado de desempates desde primera aparición **local** a
  primera aparición **global** puede cambiar una etiqueta en nodos empatados,
  incluso sin nulos. La prueba correspondiente hace explícito ese caso.
- Los tipos se deducen en entrenamiento: números son continuos, excepto booleanos;
  texto, booleanos y `category` son categóricos. Una columna numérica conserva
  ese significado al predecir aunque todos sus valores consultados sean faltantes.
- Sin datos conocidos para una columna de entrenamiento no se inventa un tipo ni
  un corte; si todo `X` está faltante, el resultado es una hoja con el prior de clases.
- Se rechazan infinitos, complejos y entradas inválidas. Se conservan los índices
  duplicados y no se modifican `X` ni `y`.
- Repartir faltantes puede multiplicar las filas representadas y el costo del árbol.
  Conviene usar prepoda positiva para experimentos con muchos nulos; los controles
  estructurales de HU-07 ya están disponibles (`docs/hu07_decisiones.md`) y la
  pospoda sigue pendiente.
- HU-12 debe comparar modelos con una política explícita de faltantes para RF y
  particiones compartidas. Esta HU no modifica RF ni acredita ejecución en Colab.

## Validación

```sh
uv run python -m unittest discover -s tests -p 'test_c45*.py' -v
```

La carga del notebook en los tests especifica UTF-8, corrigiendo el problema
detectado en la revisión del PR #2 en Windows.
`.gitattributes` conserva LF en `data/raw/`: la conversión automática a CRLF
cambiaba los bytes y hacía fallar los hashes de las copias originales.
