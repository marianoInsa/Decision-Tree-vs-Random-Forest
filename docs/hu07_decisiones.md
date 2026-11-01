# HU-07 — Controles de crecimiento y prepoda en C4.5

Implementado en las celdas de definiciones de `notebooks/01_decision_tree_c45.ipynb`.
Extiende HU-02/HU-03 y conserva el tratamiento de faltantes de HU-08. No agrega
pospoda, interfaz ni cambios en Random Forest.

## Controles

`ArbolC45(threshold=0.0, max_depth=None, min_samples_split=2, min_gain_ratio=0.0)`

| Control | Dominio | Regla de parada | Detalle |
| --- | --- | --- | --- |
| `threshold` | real finito `>= 0`, en bits | `ganancia < threshold` | Se conserva de HU-02/HU-03; la igualdad permite dividir. |
| `max_depth` | `None` o entero `>= 0` (sin `bool`) | `profundidad >= max_depth` | La raíz está en profundidad 0; `None` no limita. |
| `min_samples_split` | real finito `>= 2` (sin `bool`) | `masa < min_samples_split` | La masa es `sum(weights)`, no `len(filas)`; la igualdad permite dividir. |
| `min_gain_ratio` | real finito `>= 0` (sin `bool`) | `gain_ratio < min_gain_ratio` | Gain Ratio adimensional; la igualdad permite dividir. |

Los cuatro son decisiones del proyecto, no requisitos de C4.5 ni de una biblioteca.
En particular, usar masa efectiva en `min_samples_split` es una decisión de esta
implementación: las copias fraccionarias que genera HU-08 para filas con faltantes
no deben contarse como casos enteros nuevos.

## Orden durante el crecimiento

1. Calcular conteos, distribución y clase con la política existente.
2. Parar si la clase es pura o no quedan atributos.
3. Antes de evaluar candidatos: parar por `max_depth` y luego por masa efectiva.
4. Evaluar los cortes y quedarse con el mejor Gain Ratio, como antes.
5. Aplicar `threshold` sobre la ganancia en bits.
6. Aplicar `min_gain_ratio` sobre el Gain Ratio del mismo candidato.

`threshold` y `min_gain_ratio` son métricas distintas y **ambas** deben pasar. No se
elimina ni se reinterpreta `threshold`: sigue midiendo ganancia de información en
bits. El chequeo de ratio ocurre después de elegir el mejor candidato, de modo que
no altera el orden de selección ni los desempates.

Las hojas detenidas conservan sus conteos ponderados, su clase mayoritaria con el
desempate global de `classes_`, `predict_proba` y la inferencia con faltantes. La
poda posterior del árbol ya construido (pospoda) no forma parte de esta HU.

## Validación y compatibilidad

- La validación se ejecuta en el constructor y otra vez al comienzo de `fit`; así,
  modificar un atributo del modelo entre ambos también se rechaza.
- Se excluyen `bool`; `max_depth=2.0` se rechaza por no ser entero.
- `ArbolC45(0.05)` y `ArbolC45(threshold=0.05)` siguen funcionando: `threshold`
  continúa siendo el primer argumento posicional. Los controles nuevos pueden
  pasarse por palabra clave o en las posiciones siguientes.
- Los valores por defecto no agregan poda arbitraria, salvo la parada de masa 2.
  Esa parada puede detener nodos impuros con masa fraccionaria menor que 2 en
  escenarios de HU-08; es un efecto deliberado del control, no una regresión de
  los fixtures sin faltantes.
- No se modifican la carga de datasets, las particiones, `predict(X)` ni RF.

## Demostración y pruebas

La sección 10 del notebook compara configuraciones sobre un fixture pequeño y
muestra, con un recorrido recursivo, cómo cambian la profundidad y la cantidad de
hojas. También verifica el caso en que una copia faltante suma una fila pero no
una unidad de masa.

```sh
uv run --locked python -m unittest discover -s tests -p 'test_c45*.py' -q
```

`tests/test_c45_hu07.py` cubre validación, profundidad 0/1/None, igualdad en masa y
ratio, distinción masa/filas con faltantes, compatibilidad de `threshold` y
regresión de los fixtures existentes sin poda por defecto.

## Límites

- No se eligen hiperparámetros automáticamente ni se usa validación cruzada; eso
  corresponde a los experimentos de otra HU.
- No hay pospoda, visualización, controles interactivos ni cambios en RF.
- La métrica de ratio no reemplaza a la ganancia: un candidato puede pasar una
  condición y fallar la otra.
