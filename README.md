# Plataforma de Analisis de Sentimiento

Proyecto final del curso: plataforma para recolectar, procesar, analizar y visualizar comentarios de usuarios mediante tecnicas de procesamiento de lenguaje natural y Machine Learning.

## Objetivo

Clasificar automaticamente comentarios como `positivo`, `negativo` o `neutral`, y generar metricas utiles para la toma de decisiones.

## Funcionalidades

- Carga de comentarios desde un archivo CSV.
- Limpieza y normalizacion de texto.
- Entrenamiento de un modelo Naive Bayes multinomial.
- Prediccion de sentimiento para comentarios nuevos.
- Calculo de accuracy, tiempo de procesamiento y distribucion de sentimientos.
- Generacion de graficos con matplotlib y dashboard HTML con resultados.
- Persistencia del modelo entrenado con joblib.

## Estructura

```text
.
|-- comentarios.csv
|-- main.py
|-- README.md
|-- documento_tecnico.md
|-- presentacion.md
|-- modelo_sentimiento.joblib        # generado al entrenar
`-- resultados/
    |-- metricas.json                # generado al entrenar
    |-- dashboard.html               # generado al entrenar
    |-- distribucion_sentimientos.png
    `-- matriz_confusion.png
```

## Requisitos

Se recomienda Python 3.10 o superior.

Instalar dependencias:

```bash
pip install -r requirements.txt
```

## Ejecucion

Entrenar el modelo y generar resultados:

```bash
python main.py
```

Tambien se puede ejecutar explicitamente:

```bash
python main.py --entrenar
```

Clasificar un comentario nuevo:

```bash
python main.py --predecir "El servicio fue excelente y rapido"
```

## Dataset

El archivo `comentarios.csv` contiene comentarios etiquetados manualmente en tres categorias:

- `positivo`
- `negativo`
- `neutral`

Columnas:

- `comentario`: texto del usuario.
- `sentimiento`: etiqueta real del comentario.

## Resultados Generados

Al ejecutar el proyecto se crean:

- `modelo_sentimiento.joblib`: modelo entrenado.
- `resultados/metricas.json`: accuracy, matriz de confusion, distribucion y tiempos.
- `resultados/dashboard.html`: visualizacion de indicadores principales.
- `resultados/distribucion_sentimientos.png`: grafico de distribucion.
- `resultados/matriz_confusion.png`: grafico de matriz de confusion.

## Modelo

Se implementa un pipeline de `scikit-learn` con `TfidfVectorizer` y `MultinomialNB`. El texto se normaliza convirtiendo a minusculas, retirando signos de puntuacion, eliminando tildes y removiendo palabras vacias frecuentes en espanol. `pandas` se usa para cargar y preparar el dataset, `numpy` apoya la matriz de confusion y `matplotlib` genera los graficos.

## Alcance

La plataforma cumple con los requisitos del proyecto:

- Registro/carga de comentarios mediante CSV.
- Procesamiento automatico de texto.
- Clasificacion de sentimiento.
- Consulta de metricas.
- Visualizacion clara de resultados mediante dashboard.
