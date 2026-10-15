# Guía de ecuaciones — Decision Tree C4.5 y Random Forest

Documento de referencia para el equipo de desarrollo. Consolida las fórmulas que se usarán en
`01_decision_tree_c45.ipynb` y `02_random_forest.ipynb`, con la cita exacta de la fuente que
está en el repositorio:

- **Liu (2011)**, *Web Data Mining*, 2ª ed., Cap. 3 "Supervised Learning". El texto presenta
  las técnicas del sistema **C4.5 de Quinlan [49]**. Los números entre paréntesis, p. ej.
  (Liu, Ec. 2, p. 72), remiten a las ecuaciones numeradas de esa fuente.
- **Breiman (2001)**, *Random Forests*, Machine Learning 45, 5-32.

Todas las ecuaciones corresponden a definiciones presentes en esas dos fuentes. Las páginas
indicadas son las del texto impreso. Nota de fidelidad: los PDF no permiten copiar los
símbolos matemáticos de forma literal, por lo que las fórmulas se transcriben en notación
estándar reconstruida a partir de la prosa y de los ejemplos numéricos de las mismas fuentes.

---

## Parte I — C4.5 (Liu, Cap. 3)

### 1. Notación

Dado un conjunto de entrenamiento $D$ con $|D|$ instancias:

- $A = \{A_1, A_2, \dots, A_k\}$: conjunto de atributos disponibles.
- $C = \{c_1, c_2, \dots, c_{|C|}\}$: conjunto de clases.
- $\Pr(c_j)$: proporción de instancias de $D$ con clase $c_j$.
- La construcción del árbol es **divide-and-conquer**: en cada nodo se particiona $D$ en
  subconjuntos cada vez más puros. Un subconjunto es **puro** si todos sus ejemplos
  pertenecen a una sola clase $c_j \in C$ (Liu, p. 67).

### 2. Entropía

$$\mathrm{entropy}(D) = -\sum_{j=1}^{|C|} \Pr(c_j) \log_2 \Pr(c_j)$$

(Liu, Ec. 2, p. 72. Convención: $0 \log 0 = 0$. Unidad: bit.)

La entropía mide la impureza o desorden de $D$. Cuando el conjunto se vuelve más puro, el
valor decrece; con dos clases alcanza su máximo (1 bit) con $\Pr(c_1) = \Pr(c_2) = 0.5$ y su
mínimo (0 bit) cuando todas las instancias son de una única clase.

**Valores de verificación** (ejemplo del propio texto, p. 72):

- Con $50\%$ y $50\%$: $\mathrm{entropy}(D) = -0.5 \log_2 0.5 - 0.5 \log_2 0.5 = 1$.
- Con $20\%$ y $80\%$: $\mathrm{entropy}(D) = -0.2 \log_2 0.2 - 0.8 \log_2 0.8 = 0.722$.
- Con $100\%$ de una clase: $\mathrm{entropy}(D) = 0$.

### 3. Entropía ponderada tras la partición

Si el atributo $A_i$ tiene $v$ valores posibles $\{v_1, v_2, \dots, v_v\}$ y divide $D$ en
$v$ subconjuntos disjuntos $\{D_1, D_2, \dots, D_v\}$:

$$\mathrm{entropy}_{A_i}(D) = \sum_{j=1}^{v} \frac{|D_j|}{|D|}\, \mathrm{entropy}(D_j)$$

(Liu, Ec. 3, p. 73. Es la función `impurityEval-2` del algoritmo.)

### 4. Ganancia de información

$$\mathrm{gain}(D, A_i) = \mathrm{entropy}(D) - \mathrm{entropy}_{A_i}(D)$$

(Liu, Ec. 4, p. 73.)

La ganancia mide la reducción de impureza lograda al particionar con $A_i$. El criterio de
ganancia tiende a favorecer atributos con muchos valores posibles: un atributo
identificador (ID) produce $|D|$ subconjuntos de una instancia cada uno, con
$\mathrm{entropy}_{\text{ID}}(D) = 0$ y ganancia máxima, pero sin capacidad de
generalización (Liu, pp. 74-75).

**Ejemplo del texto con el dataset de préstamos** (p. 73): con $|D| = 15$ (6 ejemplos `No` y
9 `Yes`), $\mathrm{entropy}(D) = 0.971$, y las ganancias resultan

$$\mathrm{gain}(D, \text{CreditRating}) = 0.363 \qquad \mathrm{gain}(D, \text{OwnHouse}) = 0.420$$

eligiéndose `Own_house` como atributo raíz.

### 5. Gain ratio

Para corregir el sesgo de la ganancia, C4.5 normaliza dividiendo por la entropía de $D$
respecto de los valores del propio atributo (el denominador se denomina *split info*):

$$\mathrm{gainRatio}(D, A_i) = \frac{\mathrm{gain}(D, A_i)}{-\displaystyle\sum_{j=1}^{s} \frac{|D_j|}{|D|} \log_2 \frac{|D_j|}{|D|}}$$

(Liu, Ec. 5, pp. 74-75, donde $s$ es la cantidad de valores posibles de $A_i$ y $D_j$ el
subconjunto de datos con el $j$-ésimo valor.)

Se elige el atributo con el mayor valor de `gainRatio`. Si $A_i$ tiene demasiados valores
distintos, el denominador crece (para el atributo ID vale $\log_2 |D|$) y el atributo queda
penalizado. Nota de la fuente: el *split info* puede ser 0 o muy pequeño; para ese caso se
aplican soluciones heurísticas (Liu, p. 75).

> Criterio adoptado en el proyecto: el atributo de partición se selecciona maximizando
> **gainRatio**, no `gain`.

### 6. Partición de variables continuas

Los atributos continuos se manejan con una partición binaria: en cada nodo se busca un
umbral $t$ y $D$ se divide en los intervalos $A_i \le t$ y $A_i > t$ (Liu, pp. 75-76).

Procedimiento:

1. Sea $\{v_1, v_2, \dots, v_r\}$ el conjunto de valores distintos de $A_i$ en los datos,
   ordenado de forma ascendente.
2. Cualquier umbral entre $v_i$ y $v_{i+1}$ produce la misma partición
   $\{v_1, \dots, v_i\}$ vs. $\{v_{i+1}, \dots, v_r\}$: existen solo $r - 1$ particiones
   posibles, todas evaluables.
3. El umbral puede ser el punto medio entre $v_i$ y $v_{i+1}$, o el límite a la derecha de
   $v_i$ (esto es, $t = v_i$, con intervalos $A_i \le v_i$ y $A_i > v_i$). **C4.5 usa este
   último criterio**, de modo que los valores que aparecen en el árbol son valores reales
   observados en los datos.
4. Se selecciona el umbral que maximiza la ganancia (o el `gainRatio`).

Para cada candidato $t$:

$$\mathrm{entropy}_{A_i,t}(D) = \frac{|D_{\le t}|}{|D|}\mathrm{entropy}(D_{\le t}) + \frac{|D_{> t}|}{|D|}\mathrm{entropy}(D_{> t})$$

$$\mathrm{gain}(D, A_i, t) = \mathrm{entropy}(D) - \mathrm{entropy}_{A_i,t}(D)$$

donde $D_{\le t}$ y $D_{> t}$ son los subconjuntos definidos por el umbral. Se elige

$$t^* = \mathop{\mathrm{\arg\max}}\limits_{t} \mathrm{gain}(D, A_i, t) \quad \text{(o bien } \mathop{\mathrm{\arg\max}}\limits_{t} \mathrm{gainRatio}(D, A_i, t)\text{)}$$

(Liu, pp. 75-76.)

**Regla de retención:** para un atributo continuo, $A_g$ **no** se elimina del conjunto de
atributos al descender recursivamente, porque un intervalo se puede seguir dividiendo; así,
un mismo atributo continuo puede aparecer varias veces en un camino del árbol. Para un
atributo discreto, en cambio, $A_g$ se elimina al usarlo (Liu, pp. 75-76). El algoritmo
refleja esta regla en la llamada recursiva, comentada como "Si $A_g$ es discreto, se
elimina de $A$".

**Complejidad:** con solo atributos discretos la construcción crece linealmente con $|D|$;
ordenar un atributo continuo toma $|D| \log |D|$, costo que puede dominar el aprendizaje
del árbol (Liu, p. 76).

**Interpretación geométrica:** cada condición $A_i \le t$ es un hiperplano paralelo a un
eje; un árbol construido solo con atributos continuos particiona el espacio en
**hiper-rectángulos** alineados con los ejes (Liu, p. 76).

### 7. Algoritmo recursivo y condiciones de parada

Pseudocódigo de Liu (Fig. 3.4, p. 70), con los nombres del informe de referencia:

```text
Algoritmo decisionTree(D, A, T)
1.  SI todos los ejemplos en D pertenecen a la misma clase c_j ENTONCES
2.      Convertir T en un nodo hoja etiquetado con la clase c_j;
3.  SINO SI A = Ø ENTONCES
4.      Convertir T en un nodo hoja etiquetado con la clase c_j más frecuente en D;
5.  SINO
6.      p0 = impurityEval-1(D);
7.      PARA CADA atributo A_i en A HACER
8.          p_i = impurityEval-2(A_i, D);
9.      FIN PARA
10.     Seleccionar A_g en A que proporcione la mayor reducción de impureza (p0 - p_g);
11.     SI p0 - p_g < threshold ENTONCES   // A_g no reduce la impureza significativamente
12.         Convertir T en un nodo hoja etiquetado con la clase c_j más frecuente en D;
13.     SINO
14.         Convertir T en un nodo de decisión evaluando A_g;
15.         Sea v_1, v_2, ..., v_m los valores posibles de A_g;
16.         Particionar D en subconjuntos disjuntos D_1, D_2, ..., D_m según A_g;
17.         PARA CADA subconjunto D_j HACER
18.             SI D_j no está vacío ENTONCES
19.                 Crear una rama T_j para v_j como hijo de T;
20.                 decisionTree(D_j, A - {A_g}, T_j);  // Si A_g es discreto, se elimina de A
21.             FIN SI
22.         FIN PARA
23.     FIN SI
24. FIN SI
```

(Liu, Fig. 3.4, p. 70.)

Casos base de la recursión (líneas 1-4 y 11-13):

1. **Unanimidad de clase:** todos los registros de $D$ pertenecen a una misma clase $c_j$
   → nodo hoja puro con etiqueta $c_j$.
2. **Agotamiento de atributos:** $A = \emptyset$ pero $D$ mezcla clases → nodo hoja
   **impuro** etiquetado con la clase más frecuente en $D$.
3. **Reducción de impureza insignificante:** la mayor reducción $p_0 - p_g$ no supera un
   umbral → nodo hoja **impuro** con la clase más frecuente en $D$.

Propiedades estructurales (Liu, pp. 69-70):

- Los caminos raíz-hoja, leídos como reglas SI-ENTONCES (condiciones en los nodos de
  decisión, clase en la hoja), son **mutuamente exclusivos y exhaustivos**: cada instancia
  de entrenamiento queda cubierta por un único camino y solo uno.
- Cada hoja de un árbol construido sobre el conjunto de entrenamiento puede registrar
  $(x/y)$: $x$ de $y$ instancias que llegan a la hoja tienen la clase de la hoja; $x/y$ es
  el valor de *confidence* y $x$ el *support count* (Liu, p. 69).
- Encontrar el árbol óptimo (pequeño y preciso) es un problema **NP-completo**; por eso
  todos los algoritmos existentes usan heurísticas. El aprendizaje es *greedy* y sin
  retroceso: un nodo creado no se revisa ni se revé (Liu, pp. 69-71).

### 8. Valores faltantes (nulos)

Tratamiento nativo de C4.5 (Liu, pp. 78-79):

- **Entrenamiento:** en un nodo, las instancias con valor faltante para el atributo en
  evaluación se distribuyen a cada rama proporcionalmente a la distribución de las
  instancias que sí tienen valor para ese atributo.
- **Inferencia:** la clasificación de instancias con valores faltantes se resuelve por
  distribución probabilística sobre las ramas alcanzadas.

En nodos descendientes, los conteos de ejemplos se realizan sumando los pesos fraccionarios
de las instancias repartidas (el reparto proporcional del párrafo anterior).

### 9. Poda y control del sobreajuste

Sin poda, el árbol crece hasta eliminar la impureza y las hojas profundas terminan
respaldadas por muy pocos ejemplos: alta exactitud en entrenamiento, baja en datos no
vistos. A este fenómeno se lo denomina **sobreajuste (overfitting)** (Liu, pp. 76-77).

Métodos de poda (Liu, pp. 76-78):

- **Pre-poda (detención temprana):** interrumpe la construcción durante el entrenamiento.
  Puede detenerse de más, porque no es posible saber qué habría pasado si el árbol siguiera
  creciendo.
- **Post-poda:** primero se expande el árbol por completo y después se podan los
  subárboles/ramas que no resultan útiles. La fuente indica que la post-poda es más
  efectiva, porque con el árbol completo es más claro qué ramas sobreajustan. C4.5 sigue
  este enfoque: se estima el error de cada nodo y, si el error estimado del subárbol
  extendido no es menor que el del nodo, se poda.
- **Conjunto de validación:** alternativa de poda que usa un conjunto de datos separado
  (ni de entrenamiento ni de test); se clasifica con el árbol completo y se podan los nodos
  según los errores observados en ese conjunto.
- **Poda de reglas (rule pruning):** C4.5 convierte el árbol sin podar en un conjunto de
  reglas SI-ENTONCES y luego elimina condiciones para acortar y generalizar las reglas.
  Tras la poda, las reglas pueden dejar de ser mutuamente exclusivas y exhaustivas: puede
  haber instancias que satisfagan más de una regla o ninguna. Por eso se necesita un orden
  de las reglas y una **clase por defecto** (habitualmente la clase mayoritaria) para los
  casos que no satisfacen ninguna (Liu, p. 78).

### 10. Clases desbalanceadas

Con distribuciones de clase muy sesgadas (p. ej., intrusiones $<1\%$), el algoritmo puede
producir un árbol trivial de un solo nodo con la clase dominante. Una solución que menciona
la fuente es **sobremuestrear** los ejemplos de la clase minoritaria para aumentar su
proporción (Liu, p. 79).

### 11. Mapa función → ecuación

| Función / paso del desarrollo | Ecuación de Liu | Referencia |
| --- | --- | --- |
| Impureza del nodo | Entropía (Ec. 2) | p. 72 |
| Impureza luego de particionar (atributo discreto) | Entropía ponderada (Ec. 3) | p. 73 |
| `gain(D, A_i)` para atributos discretos | Ganancia de información (Ec. 4) | p. 73 |
| Denominador de `gainRatio` | *Split info* (denominador de Ec. 5) | pp. 74-75 |
| Criterio de selección de atributo | `gainRatio` (Ec. 5) | pp. 74-75 |
| Evaluación de umbrales de atributos continuos | Entropía ponderada + ganancia por $t$ | pp. 75-76 |
| Criterio de parada por ganancia | Línea 11 del algoritmo | p. 70 |
| Reparto de instancias con nulos | Distribución proporcional | pp. 78-79 |

---

## Parte II — Random Forest (Breiman)

### 12. Definición del ensamble

Un *random forest* es un conjunto de clasificadores de árbol $\{h(x, \Theta_k)\}$ donde los
$\Theta_k$ son vectores aleatorios independientes e idénticamente distribuidos: cada árbol
se construye con **bootstrap** de la muestra de entrenamiento y **selección aleatoria de
variables** (un subconjunto de los atributos) en cada nodo. La predicción del bosque es la
**votación por mayoría** de los árboles; el error de generalización converge a un valor
límite cuando la cantidad de árboles crece, por lo que agregar árboles no produce
sobreajuste (Breiman, Sección 2 y Sección 3, pp. 6-11).

En la variante *Forest-RI*, la cantidad $F$ de variables elegidas al azar por nodo está
fijada; los experimentos del autor usan $F = 1$ y $F = \lfloor \log_2 M + 1 \rfloor$, con
$M$ la cantidad total de variables de entrada. Los árboles se hacen crecer a tamaño máximo
y **no se podan** (Breiman, Sección 4, pp. 11-12).

### 13. Margen, strength y correlación

Función de margen del bosque (para el vector aleatorio $Y, X$):

$$mr(X, Y) = P_{\Theta}\bigl(h(X, \Theta) = Y\bigr) - \max_{j \neq Y} P_{\Theta}\bigl(h(X, \Theta) = j\bigr)$$

(Breiman, Definición 2.1, Ec. 2, p. 8.)

Strength del conjunto de clasificadores:

$$s = E_{X,Y}\, mr(X, Y)$$

(Breiman, Ec. 3, p. 8.)

Error de generalización: $PE^{*} = P_{X,Y}(mg(X, Y) < 0)$ (Breiman, p. 7). Cota superior:

$$PE^{*} \le \frac{\bar{\rho}\,(1 - s^2)}{s^2}$$

(Breiman, Teorema 2.3, p. 9, con $\bar{\rho}$ la correlación media entre los márgenes de
dos árboles cualesquiera.)

De la cota se desprende el criterio práctico del autor: el error depende de la **fuerza**
de los clasificadores individuales y de la **correlación** entre ellos; cuanto menor sea el
cociente $\bar{\rho}/s^2$ (Definición 2.4: *c/s² ratio*), mejor (Breiman, p. 9).

### 14. Out-of-bag

Cada muestra bootstrap deja afuera aproximadamente un tercio de las instancias. Agregando
solo los votos de los árboles que no vieron a una instancia se obtiene el clasificador
*out-of-bag* y su tasa de error sobre el conjunto de entrenamiento es una estimación no
sesgada del error de generalización, sin necesidad de un conjunto de test separado
(Breiman, Sección 3.1, p. 11).

---

## Referencias

- Liu, B. (2011). *Web Data Mining: Exploring Hyperlinks, Contents, and Usage Data*
  (2ª ed.), Cap. 3 "Supervised Learning", Sección 3.2 "Decision Tree Induction".
  Springer.
- Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5-32.
- Quinlan, J. R. C4.5: *Programs for Machine Learning* — sistema citado como base de la
  sección de árboles en Liu [49].
