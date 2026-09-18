# Decision Tree vs Random Forest — TP Inteligencia Artificial 2026

**Materia:** Inteligencia Artificial — 5º nivel ISI — ciclo lectivo 2026

**Integrantes:**

- Franco Sanchez
- Laura Cornaló Bassan
- Mariano Insaurralde
- Matías Fernandez
- Philippe Maurel

**[Tablero Trello](https://trello.com/invite/b/6aad9cea2c7a193807b1a25d/ATTI84c720416850d0a9b75c985ecb9163780238D185/tp-inteligencia-artificial)**

---

## Objetivo

Implementar el algoritmo **Decision Tree (DT)** íntegramente por el equipo y comparar su
rendimiento con el modelo de ensamble **Random Forest (RF)** de Scikit-learn en problemas de
clasificación, variando la parametrización de ambos modelos y midiendo métricas de desempeño.

Restricción del enunciado: el DT es 100 % propio. Para el RF sí se puede usar Scikit-learn.

---

## Estructura del repositorio

```
├── README.md
├── requirements.txt
├── data/                    # dataset indicado por la cátedra
└── notebooks/
    ├── 01_dataset.ipynb
    ├── 02_decision_tree.ipynb
    ├── 03_random_forest.ipynb
    ├── 04_experimentos.ipynb
    └── 05_interfaz.ipynb
```

---

## Etapas

Las etapas son secuenciales: `01 → 02 → 03 → 04 → 05`.
Cada notebook tiene arriba su objetivo, entradas, salidas y un checklist de tareas.

| Notebook                                             | Qué contiene                                                                         |
| ---------------------------------------------------- | ------------------------------------------------------------------------------------ |
| [01_dataset](notebooks/01_dataset.ipynb)             | Carga, exploración y preprocesamiento del dataset. Split train/test.                 |
| [02_decision_tree](notebooks/02_decision_tree.ipynb) | Implementación propia del Decision Tree y su evaluación.                             |
| [03_random_forest](notebooks/03_random_forest.ipynb) | Random Forest de Scikit-learn y su evaluación.                                       |
| [04_experimentos](notebooks/04_experimentos.ipynb)   | Comparación DT vs RF variando hiperparámetros: métricas, tiempos, tablas y gráficos. |
| [05_interfaz](notebooks/05_interfaz.ipynb)           | Interfaz amigable con ipywidgets para correr pruebas y ver resultados.               |

---

## Cómo ejecutar

### Google Colab (entrega oficial)

<-pendiente->

---

### Local

```powershell
python -m venv .venv
.venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

---

## Entregables y fechas

- Código fuente documentado que corra en Google Colab.
- Informe LNCS de Springer (≤ 12 carillas): introducción, descripción de los algoritmos,
  solución implementada, simulaciones y resultados, conclusiones y referencias.
- Tablero Trello con seguimiento de tareas.
- **Fecha límite: 11/11/2026 a las 23:59** (hasta el 18/11/2026 sin recuperatorio).
  Coloquio grupal a convenir.
