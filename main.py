import argparse
import json
import re
import time
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "comentarios.csv"
MODEL_PATH = BASE_DIR / "modelo_sentimiento.joblib"
RESULTS_DIR = BASE_DIR / "resultados"
METRICS_PATH = RESULTS_DIR / "metricas.json"
DASHBOARD_PATH = RESULTS_DIR / "dashboard.html"
CHART_PATH = RESULTS_DIR / "distribucion_sentimientos.png"
CONFUSION_PATH = RESULTS_DIR / "matriz_confusion.png"

LABELS = ["positivo", "negativo", "neutral"]
STOPWORDS = {
    "a", "al", "algo", "ante", "antes", "como", "con", "contra", "cual",
    "cuando", "de", "del", "desde", "donde", "durante", "e", "el", "ella",
    "ellas", "ellos", "en", "entre", "era", "eran", "es", "esa", "esas",
    "ese", "eso", "esos", "esta", "estaba", "estan", "estar", "estas",
    "este", "esto", "estos", "fue", "fueron", "ha", "hasta", "hay", "la",
    "las", "le", "les", "lo", "los", "mas", "me", "mi", "muy", "ni",
    "nos", "o", "para", "pero", "por", "porque", "que", "se",
    "sin", "sobre", "su", "sus", "tambien", "te", "un", "una", "unas",
    "uno", "unos", "y", "ya",
}


def limpiar_texto(texto):
    texto = str(texto).lower()
    texto = texto.translate(str.maketrans("áéíóúüñ", "aeiouun"))
    texto = re.sub(r"[^a-z0-9\s]", " ", texto)
    palabras = [p for p in texto.split() if p not in STOPWORDS and len(p) > 1]
    return " ".join(palabras)


def cargar_dataset(ruta=DATASET_PATH):
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    columnas_requeridas = {"comentario", "sentimiento"}
    if not columnas_requeridas.issubset(df.columns):
        raise ValueError("El CSV debe tener las columnas: comentario,sentimiento")

    df = df.dropna(subset=["comentario", "sentimiento"]).copy()
    df["sentimiento"] = df["sentimiento"].str.strip().str.lower()
    df = df[df["sentimiento"].isin(LABELS)]
    df["comentario_limpio"] = df["comentario"].apply(limpiar_texto)

    if df.empty:
        raise ValueError("El dataset no contiene comentarios validos.")
    return df


def crear_pipeline():
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("modelo", MultinomialNB(alpha=0.6)),
        ]
    )


def dividir_datos(df):
    return train_test_split(
        df["comentario_limpio"],
        df["sentimiento"],
        test_size=0.25,
        random_state=42,
        stratify=df["sentimiento"],
    )


def calcular_distribucion(df):
    conteos = df["sentimiento"].value_counts().reindex(LABELS, fill_value=0)
    porcentajes = (conteos / len(df) * 100).round(2)
    return {
        label: {
            "cantidad": int(conteos[label]),
            "porcentaje": float(porcentajes[label]),
        }
        for label in LABELS
    }


def generar_grafico_distribucion(distribucion):
    labels = list(distribucion.keys())
    porcentajes = [distribucion[label]["porcentaje"] for label in labels]
    colores = ["#247a4d", "#b93f32", "#6b7280"]

    plt.figure(figsize=(8, 5))
    barras = plt.bar(labels, porcentajes, color=colores)
    plt.title("Distribucion de sentimientos")
    plt.ylabel("Porcentaje")
    plt.ylim(0, max(porcentajes) + 15)

    for barra, porcentaje in zip(barras, porcentajes):
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height() + 1,
            f"{porcentaje:.2f}%",
            ha="center",
        )

    plt.tight_layout()
    plt.savefig(CHART_PATH, dpi=140)
    plt.close()


def generar_matriz_confusion(y_test, y_pred):
    matriz = confusion_matrix(y_test, y_pred, labels=LABELS)

    plt.figure(figsize=(6, 5))
    plt.imshow(matriz, cmap="Blues")
    plt.title("Matriz de confusion")
    plt.xticks(np.arange(len(LABELS)), LABELS, rotation=20)
    plt.yticks(np.arange(len(LABELS)), LABELS)
    plt.xlabel("Prediccion")
    plt.ylabel("Valor real")
    plt.colorbar()

    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            plt.text(j, i, matriz[i, j], ha="center", va="center", color="#111827")

    plt.tight_layout()
    plt.savefig(CONFUSION_PATH, dpi=140)
    plt.close()
    return matriz


def generar_dashboard(metricas, predicciones):
    filas = "\n".join(
        f"<tr><td>{fila['comentario']}</td><td>{fila['real']}</td><td>{fila['prediccion']}</td></tr>"
        for fila in predicciones
    )
    html = f"""<!doctype html>
<html lang="es">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Dashboard de Sentimientos</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; background: #f6f7f9; color: #20242a; }}
    header {{ background: #17202a; color: white; padding: 28px 40px; }}
    main {{ padding: 28px 40px; max-width: 1100px; margin: auto; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; }}
    .card {{ background: white; border: 1px solid #e1e5ea; border-radius: 8px; padding: 18px; }}
    .metric {{ font-size: 34px; font-weight: 700; margin-top: 8px; }}
    img {{ max-width: 100%; border: 1px solid #e1e5ea; border-radius: 8px; background: white; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 8px; overflow: hidden; }}
    th, td {{ text-align: left; padding: 12px; border-bottom: 1px solid #e8ebef; }}
    th {{ background: #eef1f4; }}
  </style>
</head>
<body>
  <header>
    <h1>Plataforma de Analisis de Sentimiento</h1>
    <p>Resultados generados con pandas, scikit-learn, numpy y matplotlib.</p>
  </header>
  <main>
    <section class="grid">
      <article class="card"><span>Accuracy</span><div class="metric">{metricas['accuracy']:.2%}</div></article>
      <article class="card"><span>Comentarios procesados</span><div class="metric">{metricas['total_comentarios']}</div></article>
      <article class="card"><span>Tiempo de procesamiento</span><div class="metric">{metricas['tiempo_procesamiento_segundos']}s</div></article>
    </section>
    <section class="grid" style="margin-top:16px">
      <article class="card"><h2>Distribucion de sentimientos</h2><img src="distribucion_sentimientos.png" alt="Distribucion de sentimientos"></article>
      <article class="card"><h2>Matriz de confusion</h2><img src="matriz_confusion.png" alt="Matriz de confusion"></article>
    </section>
    <section style="margin-top:16px">
      <h2>Predicciones de prueba</h2>
      <table>
        <thead><tr><th>Comentario</th><th>Real</th><th>Prediccion</th></tr></thead>
        <tbody>{filas}</tbody>
      </table>
    </section>
  </main>
</body>
</html>"""
    DASHBOARD_PATH.write_text(html, encoding="utf-8")


def entrenar():
    RESULTS_DIR.mkdir(exist_ok=True)
    inicio = time.time()
    df = cargar_dataset()
    x_train, x_test, y_train, y_test = dividir_datos(df)

    modelo = crear_pipeline()
    modelo.fit(x_train, y_train)
    y_pred = modelo.predict(x_test)
    fin = time.time()

    accuracy = accuracy_score(y_test, y_pred)
    reporte = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    matriz = generar_matriz_confusion(y_test, y_pred)
    distribucion = calcular_distribucion(df)
    generar_grafico_distribucion(distribucion)

    comentarios_originales = df.loc[x_test.index, "comentario"]
    predicciones = [
        {"comentario": comentario, "real": real, "prediccion": pred}
        for comentario, real, pred in zip(comentarios_originales, y_test, y_pred)
    ]

    metricas = {
        "accuracy": round(float(accuracy), 4),
        "total_comentarios": int(len(df)),
        "comentarios_entrenamiento": int(len(x_train)),
        "comentarios_prueba": int(len(x_test)),
        "tiempo_procesamiento_segundos": round(fin - inicio, 4),
        "distribucion": distribucion,
        "matriz_confusion": matriz.tolist(),
        "labels": LABELS,
        "classification_report": reporte,
    }

    joblib.dump(modelo, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metricas, ensure_ascii=False, indent=2), encoding="utf-8")
    generar_dashboard(metricas, predicciones)
    return metricas


def cargar_modelo():
    if not MODEL_PATH.exists():
        entrenar()
    return joblib.load(MODEL_PATH)


def predecir(comentario):
    modelo = cargar_modelo()
    comentario_limpio = limpiar_texto(comentario)
    prediccion = modelo.predict([comentario_limpio])[0]
    probabilidades = modelo.predict_proba([comentario_limpio])[0]
    return prediccion, dict(zip(modelo.classes_, probabilidades))


def main():
    parser = argparse.ArgumentParser(description="Plataforma de analisis de sentimiento.")
    parser.add_argument("--entrenar", action="store_true", help="Entrena el modelo y genera metricas.")
    parser.add_argument("--predecir", type=str, help="Clasifica un comentario nuevo.")
    args = parser.parse_args()

    if args.predecir:
        etiqueta, probabilidades = predecir(args.predecir)
        print(f"Comentario: {args.predecir}")
        print(f"Sentimiento predicho: {etiqueta}")
        print("Probabilidades:")
        for label, probabilidad in sorted(probabilidades.items()):
            print(f"- {label}: {probabilidad:.2%}")
        return

    metricas = entrenar()
    print("\n--- RESULTADOS DEL PROYECTO ---")
    print(f"Accuracy: {metricas['accuracy']:.2%}")
    print(f"Total de comentarios: {metricas['total_comentarios']}")
    print(f"Tiempo de procesamiento: {metricas['tiempo_procesamiento_segundos']} segundos")
    print("Distribucion de sentimientos:")
    for label, valores in metricas["distribucion"].items():
        print(f"- {label}: {valores['cantidad']} ({valores['porcentaje']}%)")
    print(f"\nModelo guardado en: {MODEL_PATH.name}")
    print(f"Metricas guardadas en: {METRICS_PATH}")
    print(f"Dashboard generado en: {DASHBOARD_PATH}")


if __name__ == "__main__":
    main()
