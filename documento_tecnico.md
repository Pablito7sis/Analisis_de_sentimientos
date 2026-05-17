# Documento Tecnico

## 1. Descripcion del Problema

Las organizaciones reciben comentarios de usuarios en canales como encuestas, redes sociales, formularios de soporte y tiendas en linea. Analizar manualmente esos textos puede ser lento y poco escalable. Este proyecto propone una plataforma que automatiza la clasificacion de sentimiento para identificar percepciones positivas, negativas y neutrales.

## 2. Objetivo General

Desarrollar una plataforma de analisis de datos que procese comentarios de usuarios, clasifique automaticamente su sentimiento y genere indicadores utiles para apoyar la toma de decisiones.

## 3. Objetivos Especificos

- Disenar un modelo de datos simple para almacenar comentarios y etiquetas.
- Implementar un pipeline de procesamiento de texto.
- Entrenar un modelo basico de Machine Learning para clasificacion de sentimiento.
- Crear una interfaz de consola para entrenar y predecir comentarios nuevos.
- Generar metricas y un dashboard HTML con resultados.

## 4. Arquitectura del Sistema

```text
comentarios.csv
      |
      v
Carga de datos
      |
      v
Limpieza y normalizacion de texto
      |
      v
Entrenamiento Naive Bayes
      |
      +--> modelo_sentimiento.joblib
      |
      v
Evaluacion y metricas
      |
      +--> resultados/metricas.json
      +--> resultados/dashboard.html
```

## 5. Modelo de Datos

El dataset se almacena en formato CSV con dos columnas:

| Campo | Tipo | Descripcion |
| --- | --- | --- |
| comentario | Texto | Opinion escrita por el usuario. |
| sentimiento | Categoria | Etiqueta: positivo, negativo o neutral. |

## 6. Pipeline de Procesamiento

El procesamiento aplicado a cada comentario incluye:

- Conversion a minusculas.
- Eliminacion de tildes para unificar palabras.
- Eliminacion de signos de puntuacion.
- Separacion del texto en palabras.
- Remocion de palabras vacias frecuentes en espanol.

Este pipeline reduce ruido textual y permite que el modelo se concentre en terminos con mayor valor semantico.

## 7. Explicacion del Modelo

Se utiliza un pipeline de scikit-learn compuesto por `TfidfVectorizer` y `MultinomialNB`. TF-IDF transforma los comentarios limpios en vectores numericos, asignando mayor importancia a palabras relevantes y menor peso a terminos muy frecuentes. Luego, Naive Bayes multinomial calcula la probabilidad de que un comentario pertenezca a cada clase.

Para evitar probabilidades en cero cuando aparece una palabra poco frecuente, se usa suavizado de Laplace mediante el parametro `alpha` del modelo. El modelo entrenado se guarda en `modelo_sentimiento.joblib`, lo que permite reutilizarlo para predecir comentarios nuevos.

## 8. Metricas Calculadas

El sistema genera las siguientes metricas:

- Accuracy del modelo.
- Numero total de comentarios procesados.
- Tiempo aproximado de procesamiento.
- Distribucion porcentual de sentimientos.
- Matriz de confusion.

Las metricas se guardan en `resultados/metricas.json`. Los graficos se generan con matplotlib en `resultados/distribucion_sentimientos.png` y `resultados/matriz_confusion.png`, y tambien se presentan visualmente en `resultados/dashboard.html`.

## 9. Interfaz

La interfaz principal es por consola:

- `python main.py`: entrena el modelo y genera resultados.
- `python main.py --predecir "comentario"`: clasifica un texto nuevo.

Ademas, el dashboard HTML permite consultar los resultados de forma visual.

## 10. Resultados Obtenidos

El sistema procesa el dataset incluido en `comentarios.csv`, entrena el modelo, calcula el accuracy sobre un conjunto de prueba y muestra la distribucion de sentimientos. Los resultados exactos pueden variar si se modifica el dataset, pero quedan registrados automaticamente despues de cada ejecucion.

## 11. Conclusiones

La plataforma demuestra un flujo completo de analisis de sentimiento: carga de datos, procesamiento textual, entrenamiento, evaluacion, prediccion y visualizacion. Aunque el dataset es pequeno, la arquitectura permite ampliarlo con mas comentarios para mejorar la precision del modelo y acercarlo a un escenario de mayor volumen.

## 12. Posibles Mejoras

- Aumentar el dataset con comentarios reales y mas variados.
- Crear una API REST con Flask o FastAPI.
- Incorporar modelos avanzados como regresion logistica, SVM o transformers.
- Agregar una base de datos para historico de comentarios.
- Automatizar reportes periodicos para seguimiento de tendencias.
