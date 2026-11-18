# %% 
import pandas as pd
import numpy as np
import umap
import nltk
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.cluster import HDBSCAN
from nltk.corpus import stopwords
from sklearn.metrics import mean_absolute_error
from sklearn.base import clone
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from time import perf_counter
from sklearn.model_selection import KFold


# %% 
# 0) SILENCIAMOS WARNINGS INOFENSIVOS
# warnings.filterwarnings("ignore")

# 1) CONFIG INICIAL Y DESCARGA DE PALABRAS VACÍAS (stopwords)
nltk.download("stopwords", quiet=True)
SEED = 0

# Carga de datos
df = pd.read_csv("datasets/Hotel_Reviews.csv")

# 2) LIMPIEZA DE DATOS
# Limpiamos los textos por defecto que usa el dataset por nulos
df["Negative_Review"] = df["Negative_Review"].replace("No Negative", np.nan)
df["Positive_Review"] = df["Positive_Review"].replace("No Positive", np.nan)

# Comparamos si la nota numérica del cliente cambia si la persona no deja quejas
media_sin_queja = df[df["Negative_Review"].isna()]["Reviewer_Score"].mean()
media_con_queja = df[df["Negative_Review"].notna()]["Reviewer_Score"].mean()
print(f"Puntuación media si falta queja: {media_sin_queja:.2f} | Si consta queja: {media_con_queja:.2f}")

# Rellenamos los nules con cadenas vacias para poder concatenarlas 
df["Negative_Review"] = df["Negative_Review"].fillna("")
df["Positive_Review"] = df["Positive_Review"].fillna("")

# Concatenamos las dos en una columna que se llama texto
df["texto"] = df["Negative_Review"] + " " + df["Positive_Review"]
# Quitamos espacios en blanco y si la fila esta en blanco ponemos a nulo
df["texto"] = df["texto"].str.strip().str.lower().replace("", np.nan)

df["rating"] = df["Reviewer_Score"]

# Eliminamos filas que no tengan nada de texto, no nos sirven para nada
df = df.dropna(subset=["texto", "rating"])

# Filtro: Una vez limpio el texto real, cogemos 10.000 reseñas
df = df.sample(n=10000, random_state=SEED)

# Cojo las variables X la entrada de texto e Y las estrellas que queremos predecir
X_texto = df["texto"]
y_estrellas = df["rating"]

# %%
# 3) SEPARACIÓN Y PREPARACIÓN DEL TEXTO
# Partimos los datos, el 75% para entrenar y el 25% para el examen
X_train_texto, X_test_texto, y_train, y_test = train_test_split(
    X_texto, y_estrellas,
    test_size=0.25,
    random_state=SEED
)
# Vectorización TF-IDF ignora las stopwords y se queda con las 1000 palabra mas importantes 
palabras_vacias = [
    palabra for palabra in stopwords.words("english")
    if palabra not in {"no", "nor", "not"}
]

vectorizador = TfidfVectorizer(
    stop_words=palabras_vacias,
    max_features=1000 
)

# Traducimos el texto a matriz
X_train_tfidf = vectorizador.fit_transform(X_train_texto)

print("Texto preparado!")

# %% 
# 4) REDUCCIÓN DE DIMENSIONALIDAD CON UMAP
# Reducimos las columnas a solo 10 principales
# init="random" solucion para evitar colapso de memoria
print("Iniciando la reducción de dimensionalidad, esto puede tardar algunos segundos..")
reductor_umap = umap.UMAP(
    n_components=10, 
    n_neighbors=15, 
    min_dist=0.0, 
    random_state=SEED,
    init="random",
    n_epochs=None # dejamos que umap decida las iteraciones a realizar
) 

# El reductor aprende solo de X_train_tfidf
X_train_umap = reductor_umap.fit_transform(X_train_tfidf)

# Con HDBSCAN busca grupos (clústeres) de reseñas que hablen de temas parecidos
# Cada grupo debe tener al menos 30 reseñas para ser considerado válido
agrupador = HDBSCAN(min_cluster_size=30)
etiquetas_clusters = agrupador.fit_predict(X_train_umap)

# Calculamos cuantos clústeres encontro y cuantas reseñas no encajan en nada
ruido = (etiquetas_clusters == -1).mean() # no encajan en nada -> ruido = etiqueta -1 
numero_grupos = len(set(etiquetas_clusters) - {-1})

print(f"HDBSCAN detectó {numero_grupos} grupos distintos.")
print(f"El {ruido:.0%} de las reseñas son ruido (no encajan en ningún tema principal).")

# %%
# 5) Modelos
configuraciones = [
    {"nombre": "Dummy (mediana)", "familia": "Dummy", "entrada": "base",
     "modelo": DummyRegressor(strategy="median")}
]

for alpha in [0.1, 1.0, 10.0]:
    configuraciones.append({
        "nombre": f"TF-IDF + Ridge (alpha={alpha})",
        "familia": "TF-IDF + Ridge", "entrada": "tfidf",
        "modelo": Ridge(alpha=alpha, solver="lsqr")
    })
    configuraciones.append({
        "nombre": f"UMAP + Ridge (alpha={alpha})",
        "familia": "UMAP + Ridge", "entrada": "umap",
        "modelo": Pipeline([
            ("escala", StandardScaler()),
            ("ridge", Ridge(alpha=alpha, solver="lsqr"))
        ])
    })

for C in [1.0, 10.0]:
    for epsilon in [0.1, 0.5]:
        configuraciones.append({
            "nombre": f"UMAP + SVR (C={C}, epsilon={epsilon})",
            "familia": "UMAP + SVR", "entrada": "umap",
            "modelo": Pipeline([
                ("escala", StandardScaler()),
                ("svr", SVR(kernel="rbf", C=C, epsilon=epsilon))
            ])
        })

for profundidad in [10, None]:
    configuraciones.append({
        "nombre": f"UMAP + Random Forest (profundidad={profundidad})",
        "familia": "UMAP + Random Forest", "entrada": "umap",
        "modelo": RandomForestRegressor(
            n_estimators=100, max_depth=profundidad,
            random_state=SEED, n_jobs=-1
        )
    })

pipeline_texto = Pipeline([
    ("tfidf", clone(vectorizador)),
    ("umap", clone(reductor_umap))
])

# %% 
# 6. Validación cruzada de cinco particiones

validacion = KFold(n_splits=5, shuffle=True, random_state=SEED)
resultados_cv = []

for particion, (indices_train, indices_val) in enumerate(
    validacion.split(X_train_texto), start=1):
    inicio = perf_counter()
    print(f"\nPartición {particion}/5: preparando TF-IDF y UMAP...", flush=True)

    textos_train = X_train_texto.iloc[indices_train]
    textos_val = X_train_texto.iloc[indices_val]
    notas_train = y_train.iloc[indices_train]
    notas_val = y_train.iloc[indices_val]

    # Aprendemos la preparación solo con el entrenamiento de esta partición.
    # No pasamos las notas para que UMAP siga siendo no supervisado.
    preparacion = clone(pipeline_texto)
    train_umap = preparacion.fit_transform(textos_train)
    val_umap = preparacion.transform(textos_val)

    tfidf_particion = preparacion.named_steps["tfidf"]
    entradas_train = {
        "base": np.zeros((len(textos_train), 1)),
        "tfidf": tfidf_particion.transform(textos_train),
        "umap": train_umap
    }
    entradas_val = {
        "base": np.zeros((len(textos_val), 1)),
        "tfidf": tfidf_particion.transform(textos_val),
        "umap": val_umap
    }

    # Comparamos todos los modelos con las mismas particiones.
    for configuracion in configuraciones:
        entrada = configuracion["entrada"]
        modelo = clone(configuracion["modelo"])
        modelo.fit(entradas_train[entrada], notas_train)

        # Las predicciones del regresor han de estar en el intervalo [1,10]
        predicciones = np.clip(modelo.predict(entradas_val[entrada]), 1, 10)

        mse = mean_squared_error(notas_val, predicciones)
        resultados_cv.append({
            "configuracion": configuracion["nombre"],
            "familia": configuracion["familia"],
            "entrada": entrada,
            "particion": particion,
            "MAE": mean_absolute_error(notas_val, predicciones),
            "MSE": mse,
            "RMSE": np.sqrt(mse)
        })

    print(f"Partición {particion}/5 terminada en {perf_counter() - inicio:.1f} s.",
          flush=True)

# Convertimos la lista en tabla después de completar las cinco particiones.
resultados_cv = pd.DataFrame(resultados_cv)

# Calculamos los errores medios de cada configuración.
filas_resumen = []

for configuracion in configuraciones:
    nombre = configuracion["nombre"]
    filas = resultados_cv[resultados_cv["configuracion"] == nombre]

    filas_resumen.append({
        "configuracion": nombre,
        "familia": configuracion["familia"],
        "entrada": configuracion["entrada"],
        "MSE_medio": filas["MSE"].mean(),
        "MSE_desviacion": filas["MSE"].std(),
        "RMSE_medio": filas["RMSE"].mean(),
        "RMSE_desviacion": filas["RMSE"].std()
    })

tabla_cv = pd.DataFrame(filas_resumen)
tabla_cv = tabla_cv.sort_values(["MSE_medio", "configuracion"])
tabla_cv = tabla_cv.reset_index(drop=True)

# Seleccionamos la configuración con menor MSE de cada familia.
familias = []
indices_mejores = []

for i, familia in enumerate(tabla_cv["familia"]):
    if familia not in familias:
        familias.append(familia)
        indices_mejores.append(i)

comparacion = tabla_cv.iloc[indices_mejores].reset_index(drop=True)

print("\nComparación de modelos ordenados por MSE medio:")
print(comparacion.drop(columns="entrada").round(3).to_string(index=False))

# %%
# 7) SELECCIÓN DEL MODELO Y GRÁFICAS DE MSE Y RMSE

# La primera fila corresponde al menor MSE medio en validación.
nombre_mejor = tabla_cv.iloc[0]["configuracion"]

for configuracion in configuraciones:
    if configuracion["nombre"] == nombre_mejor:
        configuracion_mejor = configuracion
        break

print(f"\nModelo seleccionado por menor MSE: {nombre_mejor}")

fig, ejes = plt.subplots(1, 2, figsize=(15, 5))

# Comparación del MSE
ejes[0].barh(
    comparacion["familia"],
    comparacion["MSE_medio"],
    xerr=comparacion["MSE_desviacion"],
    color="#2a78d6",
    capsize=4
)
ejes[0].invert_yaxis()
ejes[0].set_xlabel("MSE medio")
ejes[0].set_title("MSE de los modelos en validación")
ejes[0].grid(axis="x", alpha=0.3)

# Comparación del RMSE
ejes[1].barh(
    comparacion["familia"],
    comparacion["RMSE_medio"],
    xerr=comparacion["RMSE_desviacion"],
    color="#eb6834",
    capsize=4
)
ejes[1].invert_yaxis()
ejes[1].set_xlabel("RMSE medio")
ejes[1].set_title("RMSE de los modelos en validación")
ejes[1].grid(axis="x", alpha=0.3)

# Las barras de error representan la desviación entre particiones.
plt.tight_layout()

# %% 8) Entrenamiento y evaluación en test

entrada_mejor = configuracion_mejor["entrada"]
preparacion_final = None

if entrada_mejor == "base":
    X_train_final = np.zeros((len(X_train_texto), 1))
    X_test_final = np.zeros((len(X_test_texto), 1))
else:
    if entrada_mejor == "umap":
        preparacion_final = clone(pipeline_texto)
    else:
        preparacion_final = Pipeline([("tfidf", clone(vectorizador))])
    X_train_final = preparacion_final.fit_transform(X_train_texto)
    X_test_final = preparacion_final.transform(X_test_texto)

modelo_final = clone(configuracion_mejor["modelo"])
modelo_final.fit(X_train_final, y_train)

# Predicciones entre 1 y 10, como antes
predicciones_test = np.clip(
    modelo_final.predict(X_test_final), 1, 10
)

modelo_dummy = DummyRegressor(strategy="median")
modelo_dummy.fit(np.zeros((len(y_train), 1)), y_train)
predicciones_dummy = modelo_dummy.predict(np.zeros((len(y_test), 1)))

resultados_test = []
for nombre, predicciones in [
    (nombre_mejor, predicciones_test), ("Referencia Dummy", predicciones_dummy)
]:
    resultados_test.append({
        "modelo": nombre,
        "MAE": mean_absolute_error(y_test, predicciones),
        "RMSE": np.sqrt(mean_squared_error(y_test, predicciones)),
        "R2": r2_score(y_test, predicciones)
    })
resultados_test = pd.DataFrame(resultados_test)
print("\nEvaluación final en test:")
print(resultados_test.round(3).to_string(index=False))

mae_modelo = resultados_test.iloc[0]["MAE"]
mae_dummy = resultados_test.iloc[1]["MAE"]
if mae_modelo < mae_dummy:
    print(f"El modelo reduce el MAE del Dummy en {mae_dummy - mae_modelo:.3f} puntos.")
else:
    print("El modelo seleccionado no mejora el MAE del Dummy en este test.")

# Mostramos las gráficas de MSE y RMSE creadas en el apartado anterior
plt.show()

# Miramos qué predijo el modelo y la puntuación real
# Comparamos la puntuación real con la predicción
print ("\n")
tabla_predicciones = pd.DataFrame({
    "texto": X_test_texto.to_numpy(),
    "nota_real": y_test.to_numpy(),
    "nota_predicha": predicciones_test
})

tabla_predicciones["error_absoluto"] = np.abs(
    tabla_predicciones["nota_real"] - tabla_predicciones["nota_predicha"]
)

# Mostramos las primeras 10 predicciones
print(tabla_predicciones[
    ["nota_real", "nota_predicha", "error_absoluto"]
].head(10).round(2).to_string(index=False))


# %%
