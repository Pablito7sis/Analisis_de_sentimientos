import pandas as pd
import re
import time
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

#Descargar stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('spanish'))

#Cargar datos
df = pd.read_csv("comentarios.csv")

print("Datos cargados:")
print(df.head())

#Limpieza de texto
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r'[^a-záéíóúñ\s]', '', texto)  # quitar símbolos
    palabras = texto.split()
    palabras = [p for p in palabras if p not in stop_words]
    return " ".join(palabras)

df['comentario_limpio'] = df['comentario'].apply(limpiar_texto)

#Vectorización (texto → números)
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['comentario_limpio'])
y = df['sentimiento']

#División entrenamiento/prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Entrenamiento
modelo = MultinomialNB()

inicio = time.time()
modelo.fit(X_train, y_train)
fin = time.time()

#Predicción
y_pred = modelo.predict(X_test)

#Métricas
accuracy = accuracy_score(y_test, y_pred)

print("\n--- RESULTADOS ---")
print("Accuracy:", accuracy)
print("\nReporte:")
print(classification_report(y_test, y_pred))

print("\nTiempo de entrenamiento:", round(fin - inicio, 4), "segundos")
print("Total de comentarios:", len(df))