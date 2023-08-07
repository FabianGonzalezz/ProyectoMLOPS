# ProyectoMLOPS

Machine Learning Operations (MLOps)

# Proyecto de Análisis y Predicción de Precios de Juegos de Steam

Este proyecto tiene como objetivo realizar un análisis completo de un dataset de juegos de la plataforma Steam, incluyendo un proceso de ETL, la creación de una API utilizando FastAPI, la implementación de un modelo de machine learning para predecir los precios de los juegos y la evaluación de su rendimiento utilizando la métrica RMSE.

## Funcionalidades

El proyecto consta de las siguientes funcionalidades principales:

1. **ETL Inicial**: Se realiza una transformación de datos (ETL) en el dataset de juegos de Steam para prepararlo para su análisis posterior.

2. **API con FastAPI**: Se implementa una API web utilizando FastAPI que permite acceder a 7 funciones relacionadas con el análisis de los juegos y una para predecir el precio.

3. **ETL para Machine Learning**: Se realiza una transformación adicional en los datos para adaptarlos al proceso de entrenamiento y prueba del modelo de machine learning.

4. **Análisis Exploratorio de Datos (EDA)**: Se lleva a cabo un análisis exploratorio de datos, del cual se obtienen conclusiones clave que se presentan en forma de imágenes.

5. **Modelo de Machine Learning**: Se crea un modelo de regresión para predecir el precio de un juego en función de ciertas características.

6. **Cálculo de RMSE**: Se calcula el Root Mean Squared Error (RMSE) para evaluar el rendimiento del modelo de predicción.


## Capturas de Pantalla

Incluidas en la carpeta `src/images`:

1. [Analisis del precio promedio en relacion con el anio de salida](src/images/anio.png)
2. [Analisis del precio proomedio en relacion con el sentimiento registrado](src/images/sentimiento.png)

## Instalación y Uso

1. Clona este repositorio: `git clone https://github.com/FabianGonzalezz/ProyectoMLOPS`
2. Accede a la carpeta del proyecto: `cd PI_ML_OPS-ft`
3. Instala las dependencias: `pip install -r requirements.txt`
4. Ejecuta la API con FastAPI: `uvicorn main:app --reload`

Si tienes preguntas o comentarios, puedes contactarme en [Gmail](mailto:fabiann.m.gonzalez@gmail.com) o a través de mi perfil de GitHub: [Github](https://github.com/FabianGonzalezz/).
