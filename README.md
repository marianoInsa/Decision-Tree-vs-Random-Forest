# Árbol de decisión C4.5 y Random Forest

Trabajo práctico de Inteligencia Artificial, ciclo 2026. Integrantes: Franco Sanchez, Laura Cornaló Bassan, Mariano Insaurralde, Matías Fernandez y Philippe Maurel.

El proyecto compara un árbol propio basado en C4.5 con Random Forest de scikit-learn. Los notebooks explican sus fórmulas, decisiones, parámetros y contratos.

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

El árbol expone `entrenar`, `predecir` y `predecir_probabilidades`, con accesos compatibles `fit`, `predict` y `predict_proba`. Los cuatro controles tienen nombres en español y conservan los equivalentes `threshold`, `max_depth`, `min_samples_split` y `min_gain_ratio`. La masa mínima puede ser fraccionaria y debe ser al menos dos. `filtro_ganancia=False` conserva el criterio inicial; al activarlo se modifica la selección de divisiones y se registra como variante experimental. No hay pospoda.

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

## Google Colab y pendientes

En Colab, abrir `04_interfaz.ipynb` y ejecutar todas sus celdas. La primera celda descarga el repositorio si falta. Se utilizan las bibliotecas de Colab: no se instala `requirements.txt` completo sobre sus paquetes administrados. Ese archivo se conserva como exportación del entorno local definido en `pyproject.toml` y `uv.lock`.

La versión integrada debe comprobarse todavía en una sesión limpia de Colab. Quedan por confirmar los datos personales de la portada y la revisión editorial final. La pospoda, más particiones, validación cruzada y tiempos medidos bajo condiciones controladas son ampliaciones separadas.
