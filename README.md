# Decision Tree C4.5 vs Random Forest — TP Inteligencia Artificial 2026

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

Implementar desde cero un **Decision Tree basado en C4.5** y comparar su rendimiento con el
modelo de ensamble **Random Forest (RF)** de Scikit-learn en problemas de clasificación,
variando la parametrización de ambos modelos y midiendo métricas de desempeño.

Restricción del enunciado: el DT es 100 % propio. Para el RF sí se puede usar Scikit-learn.

---

## Estructura del repositorio

```
├── README.md
├── requirements.txt
└── notebooks/
    ├── 01_decision_tree_c45.ipynb
    ├── 02_random_forest.ipynb
    ├── 03_experimentos.ipynb
    └── 04_interfaz.ipynb
```

---

## Cómo ejecutar

### Google Colab (entrega oficial)

El notebook `notebooks/04_interfaz.ipynb` será el punto de entrada previsto para Google Colab.

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
