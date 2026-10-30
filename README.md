# Decision Tree C4.5 vs Random Forest — TP Inteligencia Artificial 2026

**Materia:** Inteligencia Artificial — 5.º nivel ISI — ciclo lectivo 2026

**Integrantes:**

- Franco Sanchez
- Laura Cornaló Bassan
- Mariano Insaurralde
- Matías Fernandez
- Philippe Maurel

**[Tablero Trello](https://trello.com/invite/b/6aad9cea2c7a193807b1a25d/ATTI84c720416850d0a9b75c985ecb9163780238D185/tp-inteligencia-artificial)**

## Objetivo

Implementar desde cero un **Decision Tree basado en C4.5** y comparar su rendimiento con
**Random Forest** de scikit-learn en problemas de clasificación, variando la parametrización
y midiendo métricas de desempeño. El Decision Tree debe ser propio; para Random Forest se
permite usar scikit-learn.

## Estructura

```text
├── data/raw/                 # Copias originales de los datasets de UCI
├── docs/
├── notebooks/
│   ├── 01_decision_tree_c45.ipynb
│   ├── 02_random_forest.ipynb
│   ├── 03_experimentos.ipynb
│   └── 04_interfaz.ipynb
├── paper/
│   └── Paper_Equipo-QG_TP_IA.pdf  # Informe final del TP
├── tests/
│   └── test_paper_build.py
├── .python-version
├── pyproject.toml
├── requirements.txt   # exportado para Google Colab
└── uv.lock
```

## Contrato común de predicción

El contrato compartido entre C4.5 y Random Forest es **`predict(X)`**. Cada modelo
se entrena previamente con su propio procedimiento. No se exige una clase propia
ni un contrato común para `fit` o `predict_proba`.

| Elemento | Definición |
|---|---|
| Entrada `X` | `pandas.DataFrame` no vacío, sin valores faltantes, con las mismas columnas, orden y significado que las características de entrenamiento. No incluye la respuesta correcta ni identificadores como características. |
| Salida | `pandas.DataFrame` con exactamente una columna llamada `prediccion` y una fila por caso de entrada. |
| Correspondencia | Conserva exactamente el índice y el orden de `X`, incluso si los índices no son consecutivos. No modifica `X`. |
| Etiquetas | Usa las mismas etiquetas de texto del dataset: las clases se detallan abajo. |
| Uso | La función queda asociada a un modelo ya entrenado. Predecir no vuelve a entrenarlo. |
| Entrada inválida | Un objeto distinto de un DataFrame produce `TypeError`; columnas incorrectas o desordenadas, datos vacíos o faltantes producen `ValueError`. |

En la preparación, **`X` e `y` son DataFrames**: `X` contiene las características
y `y` tiene una sola columna, `objetivo`. Sus índices están alineados. Para entrenar
scikit-learn se extrae internamente esa columna como vector; esto no cambia el formato
de `y` que conserva el experimento.

En `02_random_forest.ipynb`, `crear_predictor(modelo)` devuelve una función con la
firma `predict(X: pd.DataFrame) -> pd.DataFrame`. Esta función adapta la salida del
modelo de scikit-learn al contrato, sin crear otra clase de Random Forest. Para usar
un predictor entrenado, por ejemplo:

```python
predict = predictores["iris"]
predicciones = predict(particiones["iris"]["X_test"])
```

Ambos algoritmos deben compararse sobre **los mismos índices de entrenamiento y
prueba**, y con las mismas etiquetas reales. Las transformaciones particulares de
cada modelo se ajustan solo con entrenamiento. El notebook conserva los datos originales
de entrada y las particiones en `datasets` y `particiones`; el one-hot de Random Forest
se aplica dentro de su pipeline. Por ahora, un C4.5 que solo admita categorías puede
compararse directamente sobre Contact Lenses. Compararlo sobre Iris y Statlog Heart
requiere que también pueda procesar sus características numéricas.

## Datasets base

El tipo discreto, continuo o mixto describe **las características de entrada**.
Los tres problemas son de clasificación: la salida es una clase.

| Identificador | Tipo | Casos | Características | Clases y cantidades | Train / test |
|---|---|---:|---:|---|---:|
| `lenses` | Discreto / categórico | 24 | 4 | `rigidos`: 4, `blandos`: 5, `sin_lentes`: 15 | 19 / 5 |
| `iris` | Continuo | 150 | 4 | `setosa`: 50, `versicolor`: 50, `virginica`: 50 | 120 / 30 |
| `heart` | Mixto | 270 | 13 | `ausencia`: 150, `presencia`: 120 | 216 / 54 |

### Contact Lenses: discreto

Fuente: [Lenses, UCI, identificador 58](https://archive.ics.uci.edu/dataset/58/lenses),
asociado al trabajo de J. Cendrowska sobre PRISM (1987). El problema didáctico consiste
en clasificar el tipo de lentes de contacto a partir de cuatro atributos categóricos:

| Columna | Valores que recibe el modelo |
|---|---|
| `edad` | `joven`, `prepresbicia`, `presbicia` (grupos de edad, no años) |
| `prescripcion` | `miopia`, `hipermetropia` |
| `astigmatismo` | `no`, `si` |
| `lagrimas` | `reducida`, `normal` |

La clase original 1 se convierte en `rigidos`, 2 en `blandos` y 3 en `sin_lentes`.
El identificador original se usa como índice, nunca como característica. Se usa
`data/raw/lenses.data` completo: identificador, **cuatro atributos** y clase. La ficha
web de UCI presenta una discrepancia en su contador de atributos; la descripción
original conservada en `data/raw/lenses.names` documenta los cuatro, incluida la
producción de lágrimas. No hay valores faltantes.

Es útil para seguir los caminos de un árbol pequeño. Como el test tiene solo cinco
casos, cada acierto cambia la accuracy en 20 puntos porcentuales; no permite extraer
conclusiones sólidas sobre la superioridad de un algoritmo.

### Iris: continuo

Fuente: [Iris incluido en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html),
dataset clásico de Fisher. Se carga con `load_iris(as_frame=True)`, sin descarga.
Contiene las longitudes y anchuras del sépalo y del pétalo, en centímetros:
`sepal length (cm)`, `sepal width (cm)`, `petal length (cm)` y `petal width (cm)`.
El objetivo es identificar la especie: `setosa`, `versicolor` o `virginica`.
Las cuatro características son numéricas y no hay valores faltantes. Se usa la
versión distribuida por scikit-learn, que corrige dos puntos del dataset histórico;
no se mezcla con otras versiones de Iris.

### Statlog Heart: mixto

Fuente: [Statlog (Heart), UCI, identificador 145](https://archive.ics.uci.edu/dataset/145/statlog%2Bheart),
[DOI: 10.24432/C57303](https://doi.org/10.24432/C57303).
Se usa `data/raw/heart.dat`, con 270 casos y sin valores faltantes. Es la versión
**Statlog**, distinta de Heart Disease/Cleveland de 303 casos. El objetivo original
1 se convierte en `ausencia` y 2 en `presencia` de enfermedad cardíaca.

| Columna | Significado | Tratamiento |
|---|---|---|
| `edad` | Edad en años | Numérico |
| `sexo` | Código de sexo de la fuente (0/1) | Categórico |
| `dolor_pecho` | Tipo de dolor torácico (1–4) | Categórico |
| `presion_reposo` | Presión arterial en reposo | Numérico |
| `colesterol` | Colesterol sérico en mg/dl | Numérico |
| `glucemia_alta` | Glucemia en ayunas mayor a 120 mg/dl (0/1) | Categórico |
| `ecg` | Resultado electrocardiográfico (0/1/2) | Categórico |
| `frecuencia_maxima` | Frecuencia cardíaca máxima alcanzada | Numérico |
| `angina_ejercicio` | Angina inducida por ejercicio (0/1) | Categórico |
| `depresion_st` | Depresión del segmento ST inducida por ejercicio | Numérico |
| `pendiente_st` | Categoría ordinal de pendiente del segmento ST (1/2/3) | Categórico |
| `vasos` | Cantidad de vasos principales visualizados (0–3) | Numérico, conteo discreto |
| `thal` | Normal (3), defecto fijo (6) o reversible (7) | Categórico |

Los siete atributos categóricos se convierten a texto conservando sus códigos
originales. Así no se confunden con medidas continuas por estar escritos como
números en el archivo. Las seis columnas restantes conservan valores numéricos.
La descripción original está en `data/raw/heart.doc`.

### Carga, limpieza y reproducibilidad

- Las copias de UCI se conservan **sin alterar** junto con sus descripciones.
  Proceden de los archivos oficiales [lenses.zip](https://archive.ics.uci.edu/static/public/58/lenses.zip)
  y [statlog+heart.zip](https://archive.ics.uci.edu/static/public/145/statlog+heart.zip).
  Sus fichas los distribuyen bajo [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
  El notebook verifica sus hashes SHA-256 antes de cargarlos. No requiere red para
  cargar los datasets una vez instalado el entorno y disponible el repositorio.
- La preparación asigna nombres, interpreta los tipos, decodifica las clases y
  verifica tamaños, ausencia de nulos y alineación de índices. No elimina filas ni
  imputa datos: los tres archivos de entrada ya están completos.
- Cada dataset se divide con `test_size=0.20`, `random_state=42`, `shuffle=True` y
  `stratify=y["objetivo"]`. La estratificación conserva aproximadamente las
  proporciones de clases, sujeta al redondeo de los tamaños enteros.
- El modelo usa `OneHotEncoder(handle_unknown="ignore")` para categorías y deja
  pasar las columnas numéricas. El codificador se ajusta **solo con `X_train`**.
  Una categoría nueva en prueba queda representada por ceros en su grupo de columnas.
  No se normalizan escalas: el modelo base usa árboles.
- Se entrena `RandomForestClassifier` con 100 árboles, criterio Gini,
  `max_features="sqrt"`, bootstrap, profundidad sin límite, `random_state=42`
  y `n_jobs=1`. Son parámetros base, sin búsqueda de hiperparámetros ni ajuste
  usando el conjunto de prueba.
- `uv sync --locked` reproduce las dependencias fijadas en `uv.lock`. El notebook
  muestra las versiones usadas y verifica que al repetir partición y entrenamiento
  se obtengan los mismos índices y predicciones en ese entorno.

## Ejecutar Random Forest

Después de preparar el entorno local, abrir:

```sh
uv run jupyter lab notebooks/02_random_forest.ipynb
```

Reiniciar el kernel y ejecutar todas las celdas en orden. El notebook encuentra
`data/raw` desde la raíz del repositorio o desde `notebooks/`. En Colab, clonar el
repositorio completo, instalar `requirements.txt` y situar el directorio de trabajo
en la raíz del repositorio antes de ejecutar las celdas; subir solo el notebook no
incluye los archivos de datos.

La ejecución muestra tamaños y clases de las particiones, ejemplos de respuestas
reales junto con predicciones, y accuracy de prueba para los tres datasets. También
verifica el contrato de salida, la separación de train/test y la reproducibilidad.
Estas métricas documentan una ejecución base; la comparación con C4.5 queda pendiente
de integrar y ejecutar ese modelo sobre las mismas particiones.

## Entorno local con uv

Instalar `uv` siguiendo la [guía oficial](https://docs.astral.sh/uv/getting-started/installation/).
Desde la raíz del repositorio, sincronizar el entorno:

```sh
uv sync --locked
```

`uv` crea `.venv/` y selecciona Python 3.12 según `.python-version`. Si esa versión falta,
`uv` puede instalarla. Ejecutar JupyterLab dentro del entorno:

```sh
uv run jupyter lab notebooks/04_interfaz.ipynb
```

Para regenerar el archivo de dependencias usado por Colab después de cambiar dependencias:

```sh
uv export --format requirements.txt --no-dev --output-file requirements.txt
```

Editar dependencias en `pyproject.toml` con `uv add`; no editar `uv.lock` ni el `requirements.txt`
exportado manualmente.

## Paper LNCS

La carpeta `paper/` contiene la clase y el estilo bibliográfico del
[paquete oficial Springer LNCS](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines).
El archivo `.tex` deja preparada la estructura del informe, pero todavía no contiene la redacción.

Compilar desde la raíz del repositorio:

```sh
uv run python paper/build.py
```

La salida es `paper/Paper_Equipo-QG_TP_IA.pdf`. La primera compilación necesita conexión a
Internet para obtener los recursos TeX que Tectonic no tenga en caché. El script compila en un
directorio temporal, sin reutilizar auxiliares LaTeX anteriores, y comprueba el límite de 12 páginas.
El ejecutable se instala mediante [`tecto`](https://pypi.org/project/tecto/), un empaquetado
de terceros de los binarios oficiales de Tectonic.

Verificar compilación, referencias, límite de páginas y preservación de auxiliares previos:

```sh
uv run python -m unittest discover -s tests -v
```

## Google Colab

Con el repositorio disponible en el entorno de Colab, ejecutar desde su raíz:

```python
%pip install -r requirements.txt
```

Luego abrir `notebooks/04_interfaz.ipynb`. Colab aporta el entorno Jupyter; `requirements.txt`
instala las dependencias de los notebooks y se genera desde `uv.lock`.

## Entregables y fechas

- Código fuente documentado que corra en Google Colab.
- Informe LNCS de Springer de hasta 12 carillas: introducción, descripción de algoritmos,
  solución implementada, simulaciones y resultados, conclusiones y referencias.
- Tablero Trello con seguimiento de tareas.
- **Fecha límite:** 11/11/2026 a las 23:59; hasta el 18/11/2026 sin recuperatorio.
- Coloquio grupal a convenir.
