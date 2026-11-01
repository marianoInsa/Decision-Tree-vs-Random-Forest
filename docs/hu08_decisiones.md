# HU-08 — Faltantes e inferencia probabilística en C4.5

La implementación está en las celdas de definiciones de
`notebooks/01_decision_tree_c45.ipynb`. Extiende el árbol de HU-02/HU-03; no
imputa datos, no usa scikit-learn para aprender el árbol y no modifica Random Forest.

## Decisiones y recomendaciones adoptadas

| Cuestión | Recomendación implementada | Motivo |
| --- | --- | --- |
| Qué es un faltante | `None`, `NaN`, `pd.NA` y `NaT` en atributos | Cubrir las representaciones habituales de pandas sin crear una categoría artificial. |
| Etiqueta objetivo faltante | Rechazarla | No hay una clase supervisada que permita entrenar ese caso. |
| Reparto durante entrenamiento | Copiar el caso a cada rama con peso `w * q_rama`, calculado con la masa conocida del nodo | Mantener la masa total y la de cada clase, también después de varias divisiones. |
| Ganancia de información | Calcular sobre conocidos y multiplicar por su proporción de masa | Un atributo poco observado no debe recibir la misma ganancia que uno completamente observado. |
| Split Info | Incluir la masa desconocida como grupo adicional en el denominador | Los desconocidos penalizan la selección; ese grupo no se convierte en una rama. |
| Atributo vacío o constante entre los conocidos | Descartar la partición; si no queda ninguna, crear hoja con distribución local | Evitar divisiones sin información y divisiones por cero. |
| Faltante al predecir | Mezclar las distribuciones de todas las ramas según sus proporciones de entrenamiento | Conservar incertidumbre; un voto de clases mayoritarias puede dar una respuesta distinta e incorrecta. |
| Categoría nueva al predecir | Distribución local del nodo alcanzado; `predict` devuelve su mayoría | Mantener el comportamiento solicitado para categorías nuevas, sin tratarlas como faltantes. |
| Salida de `predict_proba` | DataFrame con columnas en el orden de `classes_`, índice y orden de entrada | Integración explícita con pandas y con el contrato de `predict`. |
| Orden de clases y empates de clase | Primera aparición global en `fit`; `predict` es el argmax de `predict_proba` | Evitar desempates distintos entre la predicción y las probabilidades. |
| Empates de Gain Ratio | Primer atributo por orden de columnas; menor corte numérico | Conservar el criterio determinista del árbol base. |
| Prepoda | Aplicar `threshold` a la ganancia ajustada por faltantes, manteniendo `<` | Penalizar la falta de evidencia también al decidir si dividir. |
| Precisión numérica | Conservar pesos positivos sin redondearlos; normalizar únicamente la distribución final | No eliminar de forma arbitraria ejemplos con pesos pequeños. |

Las decisiones de categorías nuevas, formato de probabilidades y orden de clases
fueron confirmadas por el responsable de la tarea. El resto son recomendaciones
de implementación documentadas para revisión del equipo.

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
  estructurales adicionales y la pospoda corresponden a HU-07.
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
