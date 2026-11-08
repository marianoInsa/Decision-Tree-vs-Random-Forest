# Integración de los aportes de los PR 6, 9, 10 y 11

La integración se prepara sobre `main` después de la reorganización de los notebooks. Se conservan las explicaciones en español, los algoritmos separados de los ejemplos, el motor común, la validación interna y la batería de faltantes. No se sustituyen los notebooks por versiones anteriores de las ramas.

| Aporte | Tratamiento |
|---|---|
| PR 6: parametrización y motor | Se conserva el motor actual, que ya cubre esa funcionalidad. No se introduce otra API de benchmarking ni pruebas que dependan de etiquetas antiguas. |
| PR 9: Cleveland | Se incorpora el archivo procesado, su descripción, el hash y la preparación común. Se actualizan experimentos, interfaz y documentación a 303 casos y seis ausencias naturales. Se retiran las copias de Statlog del árbol actual; quedan en el historial Git. |
| PR 9: imputación | Se incorpora la política explícita `imputar_frecuente`: mediana numérica y moda categórica de entrenamiento. Se conservan las políticas anteriores de rechazo y de marcador. |
| PR 10: interfaz | Se adaptan sus controles, comparaciones, barridos y dibujo al motor vigente. Se usa el esquema actual de resultados y se alinean matrices por clase antes de sumar repeticiones. |
| PR 11: filtro | Se incorpora como variante experimental desactivada por defecto, con parámetros registrados y pruebas propias. No se cambia silenciosamente el criterio inicial del árbol. |
| PR 11: experimentos e informe | Se consolidan barridos de todos los controles en el notebook 03 y se generan desde allí las tablas, figuras y observaciones del informe. No se conservan resultados numéricos históricos ni una segunda batería separada. |

Se descarta la referencia de Quinlan añadida de memoria: el filtro se describe como decisión experimental del trabajo, sin atribuirla a una obra no consultada. Las cifras de los PR anteriores no se reutilizan, porque corresponden a otras versiones del árbol y de la preparación. Los ejemplos y pruebas continúan en el notebook de experimentos, no dentro de las definiciones de los algoritmos.

La selección de configuraciones finales conserva validación interna. Los barridos de entrenamiento y prueba se presentan como descripciones de sensibilidad, sin elegir el mejor resultado de prueba como si fuera una estimación independiente. En Cleveland, cero por ciento de ausencias adicionales conserva las seis celdas naturales vacías; el porcentaje observado incluye ambas fuentes de ausencia.

Las comprobaciones locales abarcan contratos, parámetros, pesos, filtro, imputación, carga de Cleveland, botones, alineación de matrices, resultados exportados y compilación del informe. Queda pendiente ejecutar esta versión exacta en una sesión limpia de Colab. También deben confirmarse los datos personales pendientes de la portada y realizarse la revisión editorial final. El cierre de los PR indica incorporación o reemplazo de sus aportes, no una certificación de esos controles pendientes.

Validación local final: 94 pruebas aprobadas, cuatro notebooks ejecutados en kernels nuevos e informe de diez páginas compilado y revisado visualmente. El entorno y las semillas se registran en la exportación de resultados.
