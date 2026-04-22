# Análisis de Sentimientos 

## Descripción

Este proyecto consiste en el desarrollo de una plataforma capaz de analizar automáticamente el sentimiento de comentarios de usuarios utilizando técnicas de procesamiento de lenguaje natural (NLP) y Machine Learning.

El sistema permite clasificar comentarios como positivos, negativos o neutrales, generando métricas útiles para la toma de decisiones.

---

## Objetivo

Desarrollar un sistema que procese grandes volúmenes de datos textuales y determine el sentimiento de cada comentario de forma automática.

---

## Tecnologías utilizadas

* Python
* Pandas
* Scikit-learn
* NLTK

---

## Cómo ejecutar el proyecto

1. Instalar dependencias:

```
python -m pip install pandas scikit-learn nltk
```

2. Asegurarse de tener el archivo:

```
comentarios.csv
```

3. Ejecutar el script:

```
main.py
```

---

## Funcionalidades

* Carga de datos desde CSV
* Limpieza de texto
* Vectorización con TF-IDF
* Clasificación de sentimiento
* Evaluación del modelo (accuracy)

---

## Métricas generadas

* Precisión (Accuracy)
* Total de comentarios procesados
* Tiempo de entrenamiento
* Distribución de sentimientos

---

## Estructura del proyecto

```
/proyecto
│── main.py
│── comentarios.csv
│── README.md
```

---

## Estado del proyecto

Primer avance funcional: procesamiento de datos y modelo de clasificación implementado.
