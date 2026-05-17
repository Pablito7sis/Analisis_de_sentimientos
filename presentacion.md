# Presentacion: Plataforma de Analisis de Sentimiento

## 1. Titulo

Plataforma Escalable de Analisis de Sentimiento en Comentarios

## 2. Problema

Las empresas reciben muchos comentarios de usuarios y analizarlos manualmente toma tiempo. Una solucion automatica permite detectar rapidamente percepciones positivas, negativas y neutrales.

## 3. Objetivo

Crear una plataforma que procese comentarios, clasifique el sentimiento y genere indicadores visuales para apoyar la toma de decisiones.

## 4. Alcance

- Carga de comentarios desde CSV.
- Limpieza automatica de texto.
- Entrenamiento de modelo de Machine Learning.
- Prediccion de sentimiento.
- Dashboard con metricas principales.

## 5. Arquitectura

Entrada de datos con pandas -> procesamiento de texto -> TF-IDF -> modelo Naive Bayes -> metricas -> graficos y dashboard.

## 6. Dataset

El dataset contiene comentarios etiquetados como:

- Positivo
- Negativo
- Neutral

Cada registro incluye el texto del comentario y su etiqueta real.

## 7. Modelo

Se implemento un pipeline de scikit-learn con TF-IDF y Naive Bayes multinomial. El sistema convierte los comentarios en vectores numericos, calcula probabilidades por clase y selecciona la etiqueta con mayor puntaje.

## 8. Metricas

El sistema muestra:

- Accuracy.
- Total de comentarios procesados.
- Tiempo de procesamiento.
- Distribucion porcentual de sentimientos.
- Matriz de confusion.

## 9. Visualizacion

El dashboard HTML resume los indicadores principales, integra graficos generados con matplotlib y muestra ejemplos de predicciones realizadas sobre el conjunto de prueba.

## 10. Demostracion

Comandos principales:

```bash
python main.py
python main.py --predecir "El servicio fue excelente y rapido"
```

## 11. Resultados

La plataforma genera automaticamente:

- Modelo entrenado.
- Archivo de metricas.
- Dashboard visual.

## 12. Conclusiones

El proyecto cumple el flujo completo de una solucion de analisis de sentimiento y puede ampliarse con mas datos, una API REST y modelos de mayor precision.
