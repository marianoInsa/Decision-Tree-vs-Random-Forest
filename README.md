# Árbol de decisión C4.5 y Random Forest

Trabajo práctico de Inteligencia Artificial, ciclo 2026.

| Integrante | Correo | Legajo |
|---|---|---|
| Laura Cornaló Bassan | lauricornalo@gmail.com | 23354 |
| Matías Fernandez | matiasafernandez23@gmail.com | 24282 |
| Mariano Insaurralde | marianoinsaurralde5@gmail.com | 24684 |
| Philippe Maurel | philippedamau@gmail.com | 25613 |
| Franco Sanchez | francodamiansanchez10@gmail.com | 27310 |

El proyecto compara un árbol propio basado en C4.5 con Random Forest de scikit-learn. Los notebooks explican sus fórmulas, decisiones, parámetros y contratos.

[Abrir el TP de Inteligencia Artificial en Google Colab](https://colab.research.google.com/github/marianoInsa/Decision-Tree-vs-Random-Forest/blob/main/notebooks/04_interfaz.ipynb)

En Colab, seleccionar **Entorno de ejecución → Ejecutar todo**. Cuando aparezca la interfaz, usar las pestañas para comparar los modelos, variar un parámetro o dibujar el árbol.

| Archivo | Contenido |
|---|---|
| `01_decision_tree_c45.ipynb` | Definiciones del árbol: categorías, números, faltantes, prepoda y filtro opcional. |
| `02_random_forest.ipynb` | Definiciones del bosque, preparación, entrenamiento y predicción. |
| `03_experimentos.ipynb` | Ejemplos, verificaciones, validación interna, faltantes y barridos de parámetros. |
| `04_interfaz.ipynb` | Controles para comparar, variar un parámetro y visualizar el árbol. |
| `datos_tp.py` | Carga común de datos, etiquetas, tipos y particiones. |
| `paper/` | Informe LNCS y resultados regenerables. |

## Datos

| Identificador | Tipo | Casos × atributos | Clases | Ausencias naturales |
|---|---|---:|---|---|
| `lenses` | Categórico | 24 × 4 | `rigidos`: 4; `blandos`: 5; `sin_lentes`: 15 | Ninguna |
| `iris` | Continuo | 150 × 4 | Tres especies, 50 casos cada una | Ninguna |
| `heart` | Mixto | 303 × 13 | `ausencia`: 164; `presencia`: 139 | Cuatro en `vasos`, dos en `thal` |

Se usa `processed.cleveland.data` de [Heart Disease de UCI](https://archive.ics.uci.edu/dataset/45/heart+disease). Cero significa ausencia y uno a cuatro se agrupa como presencia. Se conservan las 303 filas y los signos `?` como valores ausentes. Los códigos categóricos pasan a texto. Las copias de Statlog se retiran del árbol actual para evitar confundir las dos versiones; permanecen disponibles en el historial Git.

Las copias locales de [Lenses](https://archive.ics.uci.edu/dataset/58/lenses) y Cleveland se verifican por SHA-256. Iris se obtiene de scikit-learn. No se requiere red para cargar los datos tras preparar el entorno. Los identificadores no se usan como atributos.

## Entorno y ejecución local

Desde la raíz del proyecto:

```sh
uv sync --locked
uv run --locked jupyter lab
```

Abrir `04_interfaz.ipynb` y ejecutar todas sus celdas. Los botones permiten comparar modelos, explorar un parámetro y dibujar el árbol. La interfaz carga definiciones sin ejecutar las baterías completas de experimentos.

Para dibujar el árbol en Jupyter local, instalar [Graphviz](https://graphviz.org/download/), agregar su carpeta de ejecutables al `PATH` y reiniciar Jupyter. Se puede verificar la instalación con `dot -V`. En Colab, el botón instala Graphviz si el entorno aún no lo incluye.

El árbol expone `entrenar`, `predecir` y `predecir_probabilidades`, con accesos compatibles `fit`, `predict` y `predict_proba`. Los cuatro controles tienen nombres en español y conservan los equivalentes `threshold`, `max_depth`, `min_samples_split` y `min_gain_ratio`. La masa mínima puede ser fraccionaria y debe ser al menos dos. `filtro_ganancia=False` conserva el criterio inicial. Al activarlo se modifica la selección de divisiones y se registra como variante experimental.

La pospoda está desactivada por defecto. Con `ArbolC45(pospoda=True)`, el árbol revisa su estructura de abajo hacia arriba después de entrenarse. Sustituye un subárbol por una hoja si su error estimado es igual o menor, con un factor de confianza fijo de 0,25. Usa solo los pesos y las clases del entrenamiento y conserva la distribución del nodo al convertirlo en hoja. Los resultados del informe corresponden a configuraciones sin pospoda.

En `04_interfaz.ipynb`, la casilla **Activar pospoda** se aplica a la comparación con Random Forest y al dibujo del árbol. Para comparar las dos variantes de C4.5 sobre las mismas particiones, usar **Variar un parámetro**, seleccionar **C4.5 → Pospoda** y ejecutar con `0, 1`. La tabla muestra la tasa de acierto y F1 macro en entrenamiento y prueba. Para inspeccionar el tamaño del árbol, usar **Ver el árbol** con cada estado de la casilla y conservar la semilla y la proporción de prueba.

El bosque expone `crear_bosque`, `entrenar_bosque` y `crear_predictor`; `crear_modelo` accede a la misma fábrica. Las políticas son `rechazar`, `imputar` (mediana y marcador categórico) e `imputar_frecuente` (mediana y moda). Experimentos e interfaz eligen explícitamente la última para Cleveland. Las transformaciones aprenden solo de entrenamiento.

Las entradas son DataFrames alineados; las respuestas tienen una columna `objetivo` sin faltantes. La salida tiene una columna `prediccion` y conserva índice, duplicados y orden. Se rechazan estructuras inválidas, infinitos y complejos.

## Evaluación

La referencia usa partición estratificada 80/20 y semilla 42. La selección reserva validación dentro del entrenamiento y deja prueba para la evaluación final. La batería de faltantes compara 0 %, 5 %, 15 % y 30 % adicionales con tres máscaras compartidas. En Cleveland, el 0 % adicional conserva las ausencias naturales. Se registran fracción nominal y observada.

Los barridos varían cuatro controles de crecimiento y el filtro del árbol; y cantidad de árboles, atributos candidatos, profundidad, criterio y bootstrap del bosque. Se utilizan los tres conjuntos y tres semillas de partición. Las curvas de entrenamiento y prueba son descriptivas y no seleccionan parámetros. No se presenta el mejor resultado observado en prueba como una estimación independiente.

Se calculan Accuracy, Precision, Recall y F1; las últimas tres con promedios macro y ponderados. Se guardan matrices, orden de clases, parámetros efectivos, versiones, índices y semillas. Se miden entrenamiento e inferencia por separado, excluyendo métricas y gráficos. Las matrices de distintas repeticiones se alinean por clase antes de sumarlas.

## Resultados e informe

Ejecutar completo `03_experimentos.ipynb` exporta las tablas y figuras a `paper/`. La ejecución puede tardar varios minutos. También puede utilizarse:

```sh
uv run --locked python paper/generar_resultados.py
uv run --locked python paper/build.py
```

El primer comando ejecuta el código real del notebook y genera `resultados.json`, tablas LaTeX y figuras PDF. No mantiene una segunda batería ni conserva cifras históricas. El segundo compila el informe con Tectonic en una carpeta temporal y comprueba el límite de doce páginas. Para actualizar únicamente las observaciones textuales tras ejecutar el notebook, usar `python paper/actualizar_informe.py` con el entorno preparado.

## Pruebas

```sh
uv run --locked python -X utf8 -m unittest discover -s tests -v
```

La primera compilación puede descargar los paquetes TeX. UTF-8 evita depender de la página de códigos de Windows. Las pruebas accionan los botones de la interfaz; no reemplazan una revisión visual en navegador.

## Google Colab

La primera celda de `04_interfaz.ipynb` descarga el repositorio si falta. El notebook utiliza las bibliotecas del entorno de Colab. El archivo `requirements.txt` se conserva como exportación del entorno local definido en `pyproject.toml` y `uv.lock`.
