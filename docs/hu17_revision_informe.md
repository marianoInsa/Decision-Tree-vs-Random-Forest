# Revisión técnica de Resultados y Conclusiones del informe — HU-17

Fecha: 16 de noviembre de 2026. Base revisada: `origin/main` (`942f9030bdbdb9ece20afb5a3419f2cf4aefee9b`).

## 1. Alcance de HU-17 sin duplicar trabajo previo

- **Notebook `notebooks/01_decision_tree_c45.ipynb`:** las celdas Markdown progresivas sobre Gain Ratio, prepoda y pospoda ya habían sido incorporadas en PR #13 (`657f3c4`) y `fb3eb8e`. Se conservaron intactas y se estandarizaron los cuatro docstrings que permanecían breves o ausentes (`Nodo`, `ArbolC45.__init__`, `ArbolC45._error_estimado_hoja` y `ArbolC45._podar`).
- **Informe `paper/Paper_Equipo-QG_TP_IA.tex`:** se revisaron las secciones 5 (*Simulaciones y resultados*) y 6 (*Conclusiones*) junto con las menciones al control de crecimiento en las secciones 2.1 y 4.2.

## 2. Verificación de Resultados (Sección 5)

Se contrastaron las afirmaciones de la sección 5 y de `paper/observaciones_resultados.tex` contra `paper/tablas_resultados.tex` y `docs/hu19_correspondencia_resultados.md`:

- **Conjuntos y particiones:** Lenses (24 casos, 4 categóricos), Iris (150 casos, 4 numéricos) y Cleveland procesado (303 casos, 13 atributos, 139 presencia / 164 ausencia y 6 celdas faltantes naturales). Partición estratificada 80/20 con semilla 42 y validación interna del 25 % del bloque de entrenamiento.
- **Evaluación final (`tab:finales`):**
  - Lenses: C4.5 `1.000` / `1.000` (acierto / F1 macro); Random Forest `1.000` / `1.000`.
  - Iris: C4.5 `0.933` / `0.933`; Random Forest `0.933` / `0.933`.
  - Cleveland: C4.5 `0.721` / `0.719`; Random Forest `0.836` / `0.836`.
- **Batería de faltantes (`tab:nulos` y `fig:nulos`):** 0 %, 5 %, 15 % y 30 % nominales sobre tres máscaras (semillas 42–44), conservando en 0 % las 6 ausencias naturales de Cleveland. Extremos F1 macro medio (0 % → 30 %): Lenses C4.5 `1.000 → 0.230`, RF `0.619 → 0.352`; Iris C4.5 `0.933 → 0.803`, RF `0.900 → 0.878`; Cleveland C4.5 `0.703 → 0.790`, RF `0.885 → 0.830`.
- **Filtro experimental en Cleveland:** acierto medio `0.754 ± 0.043` sin filtro y `0.738 ± 0.043` con filtro (barrido descriptivo con semillas 42–44).
- **Alcance e interpretación:** el texto distingue correctamente entre selección por validación y barridos descriptivos, advierte la sensibilidad de Lenses (5 casos en prueba) y no atribuye causalidad al tipo de atributo ni sobre-generaliza los tiempos de ejecución.

## 3. Corrección aplicada sobre pospoda (Secciones 2.1, 4.2 y 6)

El informe afirmaba en cuatro puntos (líneas 101, 154, 167 y 223 del `.tex`) que no se implementaba pospoda y la ubicaba como ampliación pendiente, contradiciendo al commit `fb3eb8e` (`ArbolC45(..., pospoda=False)`). Se actualizó la redacción para reflejar que:

1. `ArbolC45` implementa pospoda pesimista opcional (`pospoda=False` por defecto), sin implementar poda de reglas.
2. Los experimentos reportados en el informe mantuvieron la pospoda desactivada por defecto (`pospoda=False`), por lo que sus cifras permanecen vigentes sin cambios.
3. En *Conclusiones*, lo que queda como ampliación es el estudio comparativo sistemático de la pospoda, junto con la validación cruzada más extensa y las mediciones de tiempos bajo condiciones controladas.

## 4. Pendientes fuera del alcance de HU-17

Siguen abiertos como tareas de cierre editorial/operativo del equipo:
- Eventual aplicación en las secciones 2.1 y 3.1 de las observaciones **C1–C3** documentadas en `docs/revision_tecnica_informe_HU18.md`.
- Confirmación de datos personales de portada (legajo pendiente en `Paper_Equipo-QG_TP_IA.tex` y `tests/test_paper_build.py`).
- Ejecución completa en sesión limpia de Google Colab según `docs/integracion_prs.md`.

## 5. Nota de alcance posterior

Con posterioridad a esta revisión, el informe se reenfocó en Heart Disease. Se retiraron Lenses e Iris de las tablas, las observaciones y las conclusiones, y se regeneraron los resultados, las figuras y el PDF. Las cifras de las secciones 2 y 3 corresponden a la versión revisada en HU-17 y se conservan como registro histórico. La corrección sobre pospoda se mantiene en las secciones 2.1 y 6.
