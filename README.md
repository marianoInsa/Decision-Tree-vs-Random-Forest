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
