## Repositorio Entrega Final

**Facultad de Economía**

**Universidad de los Andes**

Integrantes: [David Santiago Caraballo Candela](https://github.com/scaraballoc), [Sergio David Pinilla Padilla](https://github.com/sdpinilla18) y [Juan Diego Valencia Romero](https://github.com/judval).

En el presente repositorio se encuentran todos los documentos, bases de datos y códigos utilizados durante el desarrollo del documento final la clase *Big Data & Machine Learning for Applied Economics*, del profesor [Ignacio Sarmiento-Barbieri](https://ignaciomsarmiento.github.io/igaciomsarmiento) durante el segundo semestre del año 2022.

Este trabajo tenía como objetivo el desarrollo de un modelo de predicción de individuos que incurrieran en prácticas clientelistas, a partir del uso de una base de datos del 2016 para Colombia (ELCA). Tal insumo, con la intención de mejorar el proceso de identificación de personas que hubieran respondido afirmativamente a la pregunta del Módulo de Política que indaba sobre este tipo de comportamientos. Se importa toda la *raw-database*, privada, que contiene un total de 10.000 observaciones y poco más de 800 predictores de la tercera ronda de encuesta. 

**Resumen**
¿Es posible predecir la susceptibilidad de los ciudadanos a practicar el clientelismo en las elecciones? El intercambio de votos por beneficios particularistas es esencial para comprender la relación entre la sociedad civil y el Estado. La práctica de acciones clientelistas establece una estructura de incentivos para políticos y ciudadanos que es perjudicial para el desarrollo de la capacidad estatal y el funcionamiento de la democracia. Este trabajo persigue un objetivo concreto: determinar quién venderá su voto en las próximas elecciones. Introducimos una novedosa estrategia de predicción clientelista basada en datos originales de Colombia. Nuestra métrica de evaluación corresponde a la minimización de una función de pérdida ajustada W-MIR que pondera asimétricamente los cuadrados de la tasa de falsos negativos (2/3) y falsos positivos (1/3). Encontramos que los árboles de decisión funcionan mejor de acuerdo con nuestra intención; en particular, la especificación XGBoost. Sin embargo, los modelos mejor equilibrados son los que incorporan estadística Bayesiana. 

Para poder utilizar nuestro código de **Python**, es necesario tener instalados los paquetes de `numpy`, `pyread`, `sklearn`, `pandas`, `scipy`, `contexto`, `nltk`, `spacy`, `xgboost` y `matplotlib`; de los cuales se importan diversas librerías. El código completo, que incluye todo el proceso de limpieza de datos, extracción de estadísticas descriptivas y el análisis empírico para responder a las preguntas del *problem set* se encuentran en orden dentro del notebook de Jupyter titulado "PS3_BD.ipynb". El *Python script* asociado al notebook esta titulado como "T3Script.py" y el archivo final que determina las predicciones se nombra "predictions_caraballo_pinilla_valencia.csv".

***Nota:*** *Este archivo debería correr a la perfección siempre y cuando se sigan las instrucciones y comentarios del código (en orden y forma). Es altamente recomendable que antes de verificar la replicabilidad del código, se asegure de tener **todos** los requerimientos informáticos previamente mencionados (i.e. se prefieren versiones de **Python** menores a la 3.10.9 para evitar que paquetes, funciones y métodos que han sido actualizados no funcionen). Además, la velocidad de ejecución dependerá de las características propias de su máquina, por lo que deberá (o no) tener paciencia mientras se procesa.*
