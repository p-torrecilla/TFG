# TFG: CLASIFICACIÓN DE AMENAZAS ZERO-DAY MEDIANTE APRENDIZAJE AUTOMÁTICO

## INTRODUCCIÓN

En un mundo donde el software prevalece y las amenazas abundan, es importante tener a disposición cualquier herramienta que pueda evitarnos cualquier tipo de dañoo, ya sea protegiendo información, dinero o integridad. Evitar la explotación de puertas traseras y fallos de protección es uno de los mayores desafíos que existen en el mundo de la ciberseguridad y la dificultad aumenta exponencialmente cuando se trata de amenazas zero-day. Protegerse contra amenazas desconocidas suena como un oxímoron, pero puede ser una realidad gracias a los avances en las tecnologíaas de la inteligencia artificial y el aprendizaje automático.

Con el objetivo de entrenar un clasificador mediante aprendizaje automático para identificar amenazas zero-day, se propusieron una serie de algoritmos y se procesaron los datos con distintas metodologías para obtener el mejor rendimiento de accuracy.

## DATASET

Los datos sobre los que se basaron los resultados de esta investigación fueron obtenidos de un concurso de Kaggle, el cual se encuentra en el siguiente enlace: https://www.kaggle.com/competitions/malware-detection

Estos datos se encuentran en la dirección `./malware-detection`.

## ENTORNO VIRTUAL

En `./tfgenv` se encuentra el entorno virtual que se creó para este trabajo en donde se encuentran las librerías que fueron haciendo falta. Es un entorno virtual de Python 3.13.0 y tiene algunas librerías como:
- **matplotlib**
- **lime**
- **optuna**
- **pandas**
- **sklearn**

## ALGORITMOS

Los códigos correspondientes con cada algoritmo y sus respectivas modificaciones se encuentran sobre el directorio base.

### MODELOS ELEMENTALES

Este tipo de algoritmo se utiliza como una medida sobre la cual después comparar el desempeño de los demás algoritmos y determinar si sus resultados son útiles o, por lo menos, relevantes. Se utilizaron dos versiones del clasificador "ZeroR" y un selector aleatorio.

#### ÁRBOL DE DECISIÓN

Un clasificador que se basa en una “partición binaria recursiva”, ya que el algoritmo parte los datos en subconjuntos basándose en una de las variables y, de esta manera, se generan los nodos hojas o "subárboles" sobre los que se vuelve a aplicar el mismo proceso.

#### RANDOM FOREST

Un algoritmo que está compuesto por un conjunto de árboles de decisión, los cuales llegan a clasificaciones individuales y luego cuentan los votos y se presenta una decisión final en conjunto.

### MODIFICACIONES

A los algoritmos de árbol de decisión y random forest se les realizacion ciertas modificaciones, tanto en su funcionamiento como a los datos que se les pasaron como parámetro. Estos están explicados a continuación.

#### ORIGINAL

El dataset original, sin modificar. Estos algoritmos son `random_guesser.py`, `zeroR_one.py`, `zeroR_zero.py`, `decision_tree_original.py` y `random_forest_original.py`.

#### BALANCEO

Una clase desbalanceada puede traer problemas porque le quita peso a la decisión, ya que si se opta por la clase con un peso mucho mayor, se tendrá un accuracy alto sin estar correctamente clasificando a los de la clase de menor peso.

Es por esto que se balancearon los datos y se volvieron a probar los algoritmos de árbol de decisión y random forest. Estos algoritmos son `decision_tree_balanced.py` y `random_forest_balanced.py`.

#### CORRELACIÓN

Cuando se analizan datos, a veces se encuentran características que poseen una fuerte correlación entre sí y no son necesarias a la hora de entrenar un algoritmo.

Se utilizó el coeficiente de detrminación para quedarse con con las características con correlación muy baja o despreciable, y se volvieron a ejecutar los algoritmos. Estos algoritmos son `decision_tree_correlated.py` y `random_forest_correlated.py`.

#### OPTIMIZADO

Se usó el framework *Optuna* para optimizar los clasificadores sobre la marcha. Estos algoritmos son `decision_tree_optimized.py` y `random_forest_optimized.py`.

Como estos algoritmos toman mucho tiempo en ejecutarse, se los almacenó utilizando *pickles* y se guardaron los resultados en el directorio `./optuna_algorithm_pickles`.

## RESULTADOS

Los resultados de todos estos experimentos se encuentran en el directorio `./final_results`, en donde RF representa la serie de experimentos de random forest y AD la serie de experimentos de árbol de decisión.

Además, en `./previous_results` se encuentran resultados previos de procesos intermedios o de algoritmos que no estaban correctamente configurados.

## EXTRAS

En el directorio `./non_algorithm` se encuentran otras herramientas que se utilizaron a lo largo del desarrollo del trabajo pero que no son parte de los algoritmos estrictamente. Aquí hay programas que realizan diagramas y esquemas, junto con esos esquemas producidos; estudios de correlación; relevancia de características; y más.

Aunque haya sido información relevante para el trabajo, se decidió apartarlo debido a que, de esta manera, el repositorio está más prolijo y es más fácil centrarse en los algoritmos, sus cambios y sus resultados.