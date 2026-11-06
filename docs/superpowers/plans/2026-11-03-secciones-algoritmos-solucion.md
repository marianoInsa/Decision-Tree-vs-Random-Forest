# Plan de implementación: secciones 2, 3 y 4 del informe

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Objetivo:** Redactar las secciones 2, 3 y 4 del informe LNCS con ecuaciones, un pseudocódigo, una figura de arquitectura y una tabla de decisiones, sin modificar el código de los modelos ni adelantar resultados.

**Arquitectura del documento:** La sección 2 define los conceptos y las fórmulas; la sección 3 describe los procedimientos de C4.5 y Random Forest; la sección 4 explica la organización del software y los problemas técnicos resueltos. El texto se compila con Tectonic y se verifica contra el límite de 12 páginas.

**Tecnologías:** LaTeX con la clase LNCS, Tectonic, TikZ, paquetes `algorithm` y `algorithmic`, `booktabs`, `uv`, `pypdf` y `unittest`.

**Especificación:** `AGENTS.md` contiene los lineamientos de redacción. Las fuentes técnicas son `docs/guia_algoritmos.md`, `docs/hu07_decisiones.md`, `docs/hu08_decisiones.md`, `notebooks/01_decision_tree_c45.ipynb`, `notebooks/02_random_forest.ipynb` y `datos_tp.py`.

## Restricciones globales

- Redactar en español neutro, con tercera persona o construcciones impersonales con “se”.
- Usar “trabajo” o “documento”; no usar “paper”.
- Usar “conjunto de datos”, no “dataset”.
- Escribir siempre “árbol de decisión”, “C4.5”, “Random Forest” y “scikit-learn”.
- No usar las abreviaturas “DT” ni “RF”.
- No incluir fragmentos de código ni nombres de clases, funciones o parámetros en el cuerpo.
- No nombrar los conjuntos de datos hasta la sección 5; conservar `[DATASET PENDIENTE]`.
- No adelantar resultados ni conclusiones en las secciones 2, 3 o 4.
- Evitar absolutos, superlativos y adjetivos valorativos sin evidencia.
- Presentar las comparaciones de forma equilibrada, con ventajas y costos.
- Definir cada concepto una sola vez; las secciones posteriores remiten a la definición.
- Citar a Liu para árboles de decisión y C4.5; a Breiman para Random Forest; a scikit-learn para la implementación de la biblioteca.
- Colocar las citas al final de la oración o cláusula que respaldan.
- Mantener todas las obras de `refs.bib` citadas en el texto; eliminar `\nocite` al final.
- Usar comillas ``...'' y acentos UTF-8; no usar negritas ni cursivas para destacar ideas.
- Mencionar figuras y tablas antes de que aparezcan, asignarles títulos descriptivos e interpretarlas.
- No depender del color para transmitir información.
- Mantener el informe completo en 12 páginas LNCS o menos.
- Cada sección abre con una oración que la conecta con la anterior y cierra con una transición o consecuencia.
- Cada párrafo desarrolla una idea y tiene entre cuatro y ocho oraciones.

## Foco de revisión

1. **Afirmaciones de rendimiento no respaldadas.** El texto de las secciones 2 a 4 no debe afirmar mejoras, superioridad ni resultados. Verificar con búsqueda de “mejora”, “supera”, “demuestra”, “siempre” y “nunca”; reemplazar por formulaciones prudentes o remitir a la sección 5.
2. **Identificadores del código en el cuerpo.** No deben aparecer nombres de clases, funciones, parámetros ni rutas de archivos. Verificar con una búsqueda de los identificadores del repositorio; reemplazar por prosa técnica.
3. **Exceso de alcance en valores faltantes y poda.** Random Forest no admite valores faltantes en este trabajo y no se implementa pospoda. El texto debe declarar ambas limitaciones con claridad.
4. **Definiciones duplicadas.** La entropía, la ganancia, la tasa de ganancia, el margen y la cota se definen solo en la sección 2. Las secciones 3 y 4 remiten a las ecuaciones.
5. **Desbordes y flotantes mal ubicados.** Verificar el conteo de páginas, las referencias sin resolver, los flotantes separados de su explicación y los desbordes horizontales.

## Estructura de archivos

- Modificar: `paper/Paper_Equipo-QG_TP_IA.tex`.
- Regenerar: `paper/Paper_Equipo-QG_TP_IA.pdf`.
- No modificar: `paper/refs.bib`, `paper/llncs.cls`, `paper/splncs04.bst`, `paper/build.py`.
- No modificar: código de los modelos, pruebas, cuadernos ni conjuntos de datos.
- Crear: este plan en `docs/superpowers/plans/2026-11-03-secciones-algoritmos-solucion.md`.

---

### Tarea 1: Paquetes LaTeX y sección 2.1

**Archivos:**
- Modificar: `paper/Paper_Equipo-QG_TP_IA.tex:1-54`
- Verificar: `paper/Paper_Equipo-QG_TP_IA.pdf`

**Entradas:** `docs/guia_algoritmos.md`, sección 1 a 9; `docs/hu08_decisiones.md`.
**Productos:** preámbulo con paquetes de algoritmos, TikZ y tablas; sección 2.1 redactada con ecuaciones (1) a (10).

- [ ] **Paso 1: Agregar los paquetes y la configuración de flotantes**

Después de `\usepackage{url}`, agregar:

```latex
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{tikz}
\usetikzlibrary{arrows.meta,positioning}
\usepackage{booktabs}

\floatname{algorithm}{Algoritmo}
```

Configurar las palabras clave de `algorithmic` en español dentro del preámbulo. Si alguna palabra clave no se puede traducir sin conflicto, mantener el texto algorítmico en español dentro de `\STATE` y dejar solo la estructura en inglés.

- [ ] **Paso 2: Redactar la apertura de la sección 2**

Reemplazar el comentario de `\section{Fundamentos teóricos}` por una oración que conecte con la introducción y anuncie que se definen conceptos y fórmulas. No repetir objetivos ni contribuciones.

- [ ] **Paso 3: Redactar 2.1 con ocho párrafos**

Sustituir el TODO de `\subsection{Árboles de decisión y C4.5}` por los párrafos siguientes:

1. Idea de árbol de decisión: preguntas sucesivas, nodo interno, hoja y camino como regla. Explicar “impureza” como mezcla de clases antes de usar entropía.
2. Entropía con la ecuación `eq:entropia`; convención $0\log 0=0$; interpretación como desorden.
3. Entropía ponderada `eq:entropia-particion` y ganancia `eq:ganancia`; interpretación de la ganancia.
4. Sesgo de la ganancia; entropía de la partición `eq:split-info`; tasa de ganancia `eq:gain-ratio`; caso de denominador nulo.
5. Atributos continuos: valores distintos ordenados, $r-1$ cortes, partición binaria, ganancia por umbral `eq:ganancia-umbral`; uso de valores observados; reutilización del atributo. Introducir “discretización” y aclarar que no se fija una grilla global.
6. Valores faltantes: masa total, masa conocida y masa faltante; ganancia ajustada `eq:ganancia-ajustada`; entropía de la partición con grupo desconocido `eq:split-info-ajustada`; proporciones de rama `eq:proporcion-rama`; mezcla en predicción `eq:mezcla-prediccion`.
7. Sobreajuste y poda: prepoda frente a pospoda; poda por error y poda de reglas de C4.5; carácter NP-completo del árbol óptimo; decisión de usar solo prepoda, con remisión a la sección 4.
8. Cierre con una transición a la sección 3.

Ecuaciones de referencia:

```latex
\begin{equation}
H(D) = -\sum_{c \in C} p_c \log_2 p_c
\label{eq:entropia}
\end{equation}

\begin{equation}
H_A(D) = \sum_{j=1}^{v} \frac{W_j}{W} H(D_j)
\label{eq:entropia-particion}
\end{equation}

\begin{equation}
Gain(D,A) = H(D) - H_A(D)
\label{eq:ganancia}
\end{equation}

\begin{equation}
SplitInfo(D,A) = -\sum_{j=1}^{v} \frac{W_j}{W}\log_2\frac{W_j}{W}
\label{eq:split-info}
\end{equation}

\begin{equation}
GainRatio(D,A) = \frac{Gain(D,A)}{SplitInfo(D,A)}
\label{eq:gain-ratio}
\end{equation}

\begin{equation}
Gain(D,A,t) = H(D) - \left(\frac{W_{\le t}}{W}H(D_{\le t}) + \frac{W_{>t}}{W}H(D_{>t})\right)
\label{eq:ganancia-umbral}
\end{equation}

\begin{equation}
Gain^*(D,A) = \frac{W_K}{W}\left[H(D_K) - \sum_j \frac{W_j}{W_K}H(D_j)\right]
\label{eq:ganancia-ajustada}
\end{equation}

\begin{equation}
SplitInfo^*(D,A) = -\sum_j \frac{W_j}{W}\log_2\frac{W_j}{W} - \frac{W_U}{W}\log_2\frac{W_U}{W}
\label{eq:split-info-ajustada}
\end{equation}

\begin{equation}
q_j = \frac{W_j}{W_K}
\label{eq:proporcion-rama}
\end{equation}

\begin{equation}
P(c \mid x) = \sum_j q_j P(c \mid x, \text{rama } j)
\label{eq:mezcla-prediccion}
\end{equation}
```

Agrupar las ecuaciones que comparten notación en bloques `align` si el espacio lo permite. Citar `liu2011` al final de las definiciones que correspondan.

- [ ] **Paso 4: Compilar y verificar la sección**

Ejecutar:

```bash
uv sync --locked
uv run --locked python paper/build.py
```

Resultado esperado: compilación correcta, sin referencias sin resolver, y un PDF de 12 páginas o menos.

- [ ] **Paso 5: Revisar el texto renderizado**

Ejecutar:

```bash
pdftotext -layout paper/Paper_Equipo-QG_TP_IA.pdf /tmp/opencode/paper.txt
grep -n "Fundamentos teóricos" /tmp/opencode/paper.txt
grep -n "Entropía\|entropía" /tmp/opencode/paper.txt
```

Resultado esperado: aparecen el título y los términos de la sección. Verificar que las ecuaciones no desborden el margen.

- [ ] **Paso 6: Commit**

```bash
git add paper/Paper_Equipo-QG_TP_IA.tex
git commit -m "docs(paper): redactar fundamentos teóricos de C4.5"
```

---

### Tarea 2: Sección 2.2 Random Forest

**Archivos:**
- Modificar: `paper/Paper_Equipo-QG_TP_IA.tex`, después de la sección 2.1.
- Verificar: `paper/Paper_Equipo-QG_TP_IA.pdf`.

**Entradas:** `docs/guia_algoritmos.md`, sección 12 a 14; `paper/refs.bib` y el artículo de Breiman incluido en `docs/libros/`.
**Productos:** sección 2.2 con ecuaciones (11) a (14) y transición a la sección 3.

- [ ] **Paso 1: Redactar seis párrafos**

Sustituir el TODO de `\subsection{Random Forest}` por:

1. Motivación: un árbol aislado puede cambiar ante variaciones pequeñas; el ensamble combina muchos árboles. Definir $\{h(x,\Theta_k)\}$.
2. Fuentes de aleatoriedad: muestreo con reemplazo y subconjunto aleatorio de atributos por corte; árboles profundos sin poda.
3. Predicción por voto mayoritario `eq:voto-mayoritario`.
4. Margen `eq:margen`, strength `eq:strength` y cota `eq:cota`; interpretar fuerza y correlación. Aclarar que la cota explica de qué depende el error, no que se use para calcularlo.
5. Muestras fuera de la bolsa: aproximadamente un tercio queda sin seleccionar; estimación del error sin conjunto separado.
6. Configuración de scikit-learn: cien árboles, criterio de Gini, raíz cuadrada del número de atributos por corte, muestreo con reemplazo y sin límite de profundidad. Cerrar con una transición a la sección 3.

Ecuaciones de referencia:

```latex
\begin{equation}
\hat{y}(x) = \arg\max_{c} \sum_{k=1}^{B} \mathbf{1}[h_k(x)=c]
\label{eq:voto-mayoritario}
\end{equation}

\begin{equation}
mr(x,y) = P_\Theta(h(x,\Theta)=y) - \max_{j\ne y} P_\Theta(h(x,\Theta)=j)
\label{eq:margen}
\end{equation}

\begin{equation}
s = E_{X,Y}[mr(X,Y)]
\label{eq:strength}
\end{equation}

\begin{equation}
PE^* \le \frac{\bar{\rho}(1-s^2)}{s^2}
\label{eq:cota}
\end{equation}
```

- [ ] **Paso 2: Verificar terminología y citas**

Buscar que “Random Forest”, “muestreo con reemplazo”, “muestras fuera de la bolsa” y “tasa de ganancia” se usen de forma consistente. Confirmar que `breiman2001` respalda las definiciones y que `scikitRF` respalda la configuración de la biblioteca.

- [ ] **Paso 3: Compilar y revisar**

```bash
uv run --locked python paper/build.py
pdftotext -layout paper/Paper_Equipo-QG_TP_IA.pdf /tmp/opencode/paper.txt
grep -n "Random Forest" /tmp/opencode/paper.txt
```

Resultado esperado: sección completa, ecuaciones numeradas y sin desbordes.

- [ ] **Paso 4: Commit**

```bash
git add paper/Paper_Equipo-QG_TP_IA.tex
git commit -m "docs(paper): redactar fundamentos teóricos de Random Forest"
```

---

### Tarea 3: Sección 3 y Algoritmo 1

**Archivos:**
- Modificar: `paper/Paper_Equipo-QG_TP_IA.tex`, sección 3.
- Verificar: `paper/Paper_Equipo-QG_TP_IA.pdf`.

**Entradas:** `notebooks/01_decision_tree_c45.ipynb`, clase `ArbolC45` y funciones de métricas; `docs/hu07_decisiones.md`; `docs/hu08_decisiones.md`.
**Productos:** sección 3 con el pseudocódigo de C4.5 y la descripción de Random Forest; ninguna definición nueva de entropía, ganancia o tasa de ganancia.

- [ ] **Paso 1: Redactar la apertura y 3.1**

Reemplazar el TODO de `\section{Breve descripción de los algoritmos desarrollados}` por una oración de transición. Luego, en `\subsection{C4.5}`, escribir cinco párrafos:

1. Estrategia recursiva de dividir y conquistar; entradas y salida en prosa.
2. Presentación del Algoritmo 1, mencionado antes del flotante.
3. Explicación de las decisiones del pseudocódigo: conteos ponderados, paradas, evaluación de candidatos, selección por tasa de ganancia, comparación estricta de umbrales, desempates.
4. Predicción: descenso por valor conocido, mezcla de ramas ante faltante, distribución local ante categoría nueva, argmax y desempate global. Remitir a `eq:mezcla-prediccion`.
5. Costo y límites: ordenar atributos numéricos en cada nodo, cantidad de cortes candidatos, crecimiento codicioso sin retroceso y ausencia de pospoda.

- [ ] **Paso 2: Escribir el Algoritmo 1**

Usar `\begin{algorithm}` y `\begin{algorithmic}`. Título: “Construcción recursiva de un árbol C4.5”. Sin nombres de clases, funciones ni parámetros del código. Pasos:

1. Calcular la masa por clase y crear una hoja con la clase mayoritaria.
2. Devolver la hoja si hay una sola clase, si no quedan atributos, si se alcanzó la profundidad máxima o si la masa es menor que el mínimo.
3. Inicializar el mejor candidato como vacío.
4. Para cada atributo: si es numérico, evaluar cada valor observado salvo el máximo como umbral; si es categórico, evaluar una rama por categoría conocida.
5. Calcular la tasa de ganancia ajustada por valores faltantes.
6. Conservar el candidato con mayor tasa de ganancia; desempatar por orden de atributo y menor umbral.
7. Devolver la hoja si no hay candidato o si no se alcanzan el umbral de ganancia y la tasa mínima.
8. Crear el nodo de decisión y registrar el atributo, el umbral o las ramas, y las proporciones de cada rama.
9. Para cada rama, formar el subconjunto con los casos conocidos y repartir los faltantes con peso proporcional.
10. Eliminar el atributo usado si es categórico; conservarlo si es numérico.
11. Construir recursivamente cada hijo y devolver el nodo.

- [ ] **Paso 3: Redactar 3.2**

En `\subsection{Random Forest}`, escribir tres párrafos:

1. Procedimiento: $B$ muestras con reemplazo, un árbol por muestra, subconjunto aleatorio de atributos en cada corte, crecimiento profundo y voto mayoritario.
2. Diferencia con el árbol único: la aleatoriedad reduce la correlación entre árboles; el voto combina respuestas. No afirmar mejoras empíricas.
3. Costo e interpretabilidad: el entrenamiento se multiplica por la cantidad de árboles; la decisión individual deja de seguirse en una regla única. Cerrar con una transición a la sección 4.

- [ ] **Paso 4: Verificar que no haya código ni definiciones duplicadas**

```bash
grep -nE "ArbolC45|predict_proba|predict\(|fit\(|min_samples_split|min_gain_ratio|max_depth|threshold|n_estimators|max_features" paper/Paper_Equipo-QG_TP_IA.tex
```

Resultado esperado: ninguna coincidencia en el cuerpo. Reemplazar cualquier identificador por una descripción en prosa.

- [ ] **Paso 5: Compilar y revisar el flotante**

```bash
uv run --locked python paper/build.py
pdftotext -layout paper/Paper_Equipo-QG_TP_IA.pdf /tmp/opencode/paper.txt
grep -n "Algoritmo 1\|Construcción recursiva" /tmp/opencode/paper.txt
```

Resultado esperado: el algoritmo aparece completo, cerca de su explicación y sin desbordes.

- [ ] **Paso 6: Commit**

```bash
git add paper/Paper_Equipo-QG_TP_IA.tex
git commit -m "docs(paper): describir algoritmos y agregar pseudocódigo de C4.5"
```

---

### Tarea 4: Sección 4.1 y Figura 1

**Archivos:**
- Modificar: `paper/Paper_Equipo-QG_TP_IA.tex`, sección 4.
- Verificar: `paper/Paper_Equipo-QG_TP_IA.pdf`.

**Entradas:** `datos_tp.py`, notebooks 01 y 02, `tests/`, `paper/build.py`.
**Productos:** título “Descripción de la solución implementada”, sección 4.1 y figura de arquitectura.

- [ ] **Paso 1: Cambiar el título de la sección 4**

Reemplazar `\section{Solución implementada}` por `\section{Descripción de la solución implementada}`.

- [ ] **Paso 2: Redactar la apertura y 4.1**

En `\subsection{Arquitectura de software}`, escribir:

1. Organización general en tres capas: preparación de datos, modelos y evaluación. Mencionar la Figura 1 antes del flotante.
2. Figura 1 con el flujo de datos y las dos ramas de modelado.
3. Preparación: verificación de integridad, decodificación de etiquetas, distinción entre atributos numéricos y categóricos, partición estratificada con semilla fija; sin imputación ni eliminación de filas.
4. Modelos: C4.5 propio con funciones de métricas, nodo y construcción recursiva; Random Forest con un pipeline de scikit-learn y codificación binaria de categorías solo dentro del pipeline. Explicar por qué la codificación no se aplica a C4.5.
5. Contrato común: la operación de predicción recibe un DataFrame y devuelve una columna con la clase, conserva índice y orden; no se unifican el entrenamiento ni las clases. Explicar el bajo acoplamiento y la comparación justa.
6. Evaluación y reproducibilidad: mismas particiones, semilla fija, desempates deterministas, un solo hilo en Random Forest y pruebas que cargan las celdas de los cuadernos para evitar duplicación.
7. Cierre con una transición a las decisiones de diseño.

- [ ] **Paso 3: Dibujar la Figura 1 con TikZ**

Figura en blanco y negro, sin depender del color. Nodos:

- “Conjuntos de datos originales”
- “Carga y validación”
- “Partición estratificada”
- “Entrenamiento” y “Prueba”
- “Árbol C4.5” y “Random Forest”
- “Predicción común”
- “Evaluación”

Aristas: datos originales → carga → partición; partición → entrenamiento y prueba; entrenamiento → ambos modelos; ambos modelos → predicción común; predicción común y prueba → evaluación. Anotar “mismas particiones” junto a las ramas de entrenamiento y prueba. Epígrafe: “Organización de la solución implementada.”

- [ ] **Paso 4: Compilar y revisar la figura**

```bash
uv run --locked python paper/build.py
pdftotext -layout paper/Paper_Equipo-QG_TP_IA.pdf /tmp/opencode/paper.txt
grep -n "Organización de la solución" /tmp/opencode/paper.txt
```

Resultado esperado: la figura aparece después de su mención, con texto legible y sin desbordes.

- [ ] **Paso 5: Commit**

```bash
git add paper/Paper_Equipo-QG_TP_IA.tex paper/Paper_Equipo-QG_TP_IA.pdf
git commit -m "docs(paper): describir arquitectura y agregar figura de la solución"
```

---

### Tarea 5: Secciones 4.2, 4.3 y Tabla 1

**Archivos:**
- Modificar: `paper/Paper_Equipo-QG_TP_IA.tex`, sección 4.
- Verificar: `paper/Paper_Equipo-QG_TP_IA.pdf`.

**Entradas:** `docs/hu07_decisiones.md`, `docs/hu08_decisiones.md`, README, pruebas de C4.5.
**Productos:** decisiones de diseño, problemas técnicos resueltos y tabla de síntesis.

- [ ] **Paso 1: Redactar 4.2 con cuatro párrafos**

1. Particiones compartidas y preprocesamiento específico de cada modelo; la codificación binaria de categorías se ajusta solo con entrenamiento.
2. Contrato mínimo de predicción; conservación de índice y orden; validación de entrada; ausencia de mutación.
3. Tipos de atributos y desempates; deducción en entrenamiento; corte en valores observados; orden estable de clases.
4. Semántica de las probabilidades: frecuencias ponderadas, sin calibración; argmax para la etiqueta; mezcla de ramas ante faltantes y distribución local ante categoría nueva. Cerrar con una transición a los problemas técnicos.

- [ ] **Paso 2: Redactar 4.3 con cinco párrafos y la tabla**

1. Introducción: tres dificultades principales y la integración reproducible.
2. Atributos continuos y discretización: problema de una grilla global; decisión de evaluar cortes binarios por nodo en valores observados; costo de ordenar; reutilización; consecuencia sobre interpretabilidad y sobreajuste.
3. Valores faltantes: problema de la imputación; pesos fraccionarios, ganancia ajustada, grupo desconocido en la entropía de la partición y mezcla de distribuciones; conservación de masa; masa efectiva para la parada; atributo constante o totalmente vacío; diferencia entre faltante y categoría nueva. Aclarar que Random Forest requiere datos completos en este trabajo.
4. Crecimiento y poda: problema del sobreajuste; prepoda con cuatro controles independientes; umbral de ganancia en bits, profundidad máxima, masa mínima y tasa de ganancia mínima; comparación estricta; ambas condiciones de ganancia deben cumplirse. Explicitar que no se implementa pospoda ni poda de reglas y que la prepoda puede detener el crecimiento antes de lo conveniente.
5. Integración y reproducibilidad: una sola implementación; pruebas que cargan las celdas de los cuadernos; verificación de integridad; conservación de saltos de línea; resolución de rutas en distintos entornos.
6. Mencionar la Tabla 1 antes de aparecer. Título: “Problemas técnicos, decisiones adoptadas y verificación”. Columnas: problema, decisión, verificación. Filas: atributos continuos, valores faltantes, crecimiento, reproducibilidad. Interpretar la tabla en un párrafo y cerrar con una transición a la sección 5.

Contenido de la tabla:

| Problema | Decisión | Verificación |
|---|---|---|
| Atributos continuos | Cortes binarios por nodo en valores observados; sin discretización global | Pruebas con cortes conocidos y reutilización de atributos |
| Valores faltantes | Pesos fraccionarios, ganancia ajustada y mezcla de distribuciones | Cálculo manual y conservación de masa |
| Crecimiento | Prepoda con cuatro controles independientes; sin pospoda | Pruebas de profundidad, masa y tasa de ganancia |
| Reproducibilidad | Semilla fija, verificación de integridad y desempates deterministas | Repetición de particiones y predicciones |

- [ ] **Paso 3: Verificar límites y afirmaciones**

```bash
grep -niE "dataset|accuracy|\bDT\b|\bRF\b|mejora|supera|demuestra|siempre|nunca|excelente|poderoso" paper/Paper_Equipo-QG_TP_IA.tex
```

Resultado esperado: ninguna coincidencia en el cuerpo. Revisar las excepciones justificadas.

- [ ] **Paso 4: Compilar y revisar la tabla**

```bash
uv run --locked python paper/build.py
pdftotext -layout paper/Paper_Equipo-QG_TP_IA.pdf /tmp/opencode/paper.txt
grep -n "Problemas técnicos" /tmp/opencode/paper.txt
```

Resultado esperado: tabla completa, mencionada antes y sin desbordes.

- [ ] **Paso 5: Commit**

```bash
git add paper/Paper_Equipo-QG_TP_IA.tex paper/Paper_Equipo-QG_TP_IA.pdf
git commit -m "docs(paper): describir decisiones y problemas técnicos"
```

---

### Tarea 6: Revisión global, consistencia y cierre

**Archivos:**
- Modificar: `paper/Paper_Equipo-QG_TP_IA.tex` si la revisión detecta problemas.
- Regenerar: `paper/Paper_Equipo-QG_TP_IA.pdf`.
- Verificar: `tests/test_paper_build.py`.

**Entradas:** todo el documento, `AGENTS.md`, `paper/refs.bib`.
**Productos:** documento compilado, sin referencias sin resolver, con citas completas y dentro del límite de páginas.

- [ ] **Paso 1: Revisar la introducción existente**

Leer la introducción y corregir solo lo necesario para cumplir los lineamientos:

- Reemplazar “la precisión mejora” por una formulación prudente, por ejemplo “puede mejorar la precisión”, con la cita correspondiente.
- Reemplazar “un árbol aislado es sensible” por “un árbol aislado suele ser sensible”.
- Confirmar que no aparecen “paper”, “dataset”, “DT”, “RF” ni nombres de conjuntos de datos.
- Confirmar que la hoja de ruta final sigue siendo correcta después de cambiar el título de la sección 4.

- [ ] **Paso 2: Revisar citas y bibliografía**

Confirmar que `liu2011`, `breiman2001` y `scikitRF` están citados en el texto. Eliminar `\nocite{liu2011,breiman2001,scikitRF}` si las tres citas existen. Compilar y verificar que la bibliografía conserva el estilo `splncs04`.

- [ ] **Paso 3: Verificar que no haya código ni nombres de conjuntos de datos**

```bash
grep -nE "ArbolC45|predict_proba|predict\(|fit\(|min_samples_split|min_gain_ratio|max_depth|threshold|n_estimators|max_features" paper/Paper_Equipo-QG_TP_IA.tex
grep -niE "lenses|iris|heart" paper/Paper_Equipo-QG_TP_IA.tex
```

Resultado esperado: ninguna coincidencia en el cuerpo.

- [ ] **Paso 3b: Verificar las limitaciones y la ausencia de definiciones duplicadas**

Confirmar que el texto declara que Random Forest requiere datos completos y que no se implementa pospoda ni poda de reglas. Confirmar que la entropía, la ganancia, la tasa de ganancia, el margen y la cota se definen solo en la sección 2; las secciones 3 y 4 deben remitir a esas definiciones sin repetirlas.

```bash
grep -n "pospoda\|poda de reglas\|datos completos\|valores faltantes" paper/Paper_Equipo-QG_TP_IA.tex
```

Resultado esperado: las limitaciones aparecen una vez y no hay definiciones repetidas.

- [ ] **Paso 4: Compilar y verificar el límite de páginas**

```bash
uv run --locked python paper/build.py
```

Resultado esperado: `Generated ...pdf (N pages)` con `N` menor o igual que 12. Si se supera el límite, recortar primero ejemplos y repeticiones, no definiciones ni ecuaciones.

- [ ] **Paso 5: Ejecutar la suite completa**

```bash
uv run --locked python -m unittest discover -s tests -v
```

Resultado esperado: todas las pruebas pasan, incluida la compilación del informe y la verificación de autores y referencias.

- [ ] **Paso 6: Revisión visual del PDF**

Abrir `paper/Paper_Equipo-QG_TP_IA.pdf` y verificar:

- Títulos de sección y subsección con el estilo LNCS.
- Ecuaciones numeradas y referenciadas.
- Algoritmo 1 completo y cercano a su explicación.
- Figura 1 legible y en blanco y negro.
- Tabla 1 con título superior y sin desbordes.
- Sin flotantes al final del documento ni páginas con líneas desbordadas.
- Sin “??” en referencias o citas.

- [ ] **Paso 7: Commit final**

```bash
git add paper/Paper_Equipo-QG_TP_IA.tex paper/Paper_Equipo-QG_TP_IA.pdf
git commit -m "docs(paper): completar secciones de algoritmos y solución"
```

---

## Autoevaluación del plan

- Cobertura: secciones 2, 3 y 4; ecuaciones, pseudocódigo, figura y tabla; estilo y citas.
- Fuera de alcance: resumen, sección 5, conclusiones, interfaz, cambios de código y nuevos conjuntos de datos.
- Riesgo principal: el límite de 12 páginas. La estimación es de 5,5 a 6 páginas para las secciones nuevas, con margen para resultados y conclusiones.
- Dependencia externa: disponibilidad de TikZ y de los paquetes de algoritmos en Tectonic. Si TikZ falla, la figura se reemplaza por un diagrama tabular en blanco y negro.
