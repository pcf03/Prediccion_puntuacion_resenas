import pandas as pd
import numpy as np
import umap
import nltk
import warnings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.cluster import HDBSCAN
from nltk.corpus import stopwords

# 0) SILENCIAMOS WARNINGS INOFENSIVOS
warnings.filterwarnings("ignore")

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

# Pasamos de 1-10 estrellas a 1-5 para la fase 2
df["rating"] = np.ceil(df["Reviewer_Score"] / 2)

# Eliminamos filas que no tengan nada de texto, no nos sirven para nada
df = df.dropna(subset=["texto", "rating"])

# Filtro: Una vez limpio el texto real, cogemos 100.000 reseñas al azar
df = df.sample(n=10000, random_state=SEED)

# Cojo las variables X la entrada de texto e Y las estrellas que queremos predecir
X_texto = df["texto"]
y_estrellas = df["rating"]


# 3) SEPARACIÓN Y PREPARACIÓN DEL TEXTO
# Partimos los datos, el 75% para entrenar y el 25% para el examen
X_train_texto, X_test_texto, y_train, y_test = train_test_split(
    X_texto, y_estrellas, test_size=0.25, random_state=SEED, stratify=y_estrellas
)

# Vectorización TF-IDF ignora las stopwords y se queda con las 1000 palabra mas importantes para que no sature la RAM
vectorizador = TfidfVectorizer(
    stop_words=stopwords.words("english"),
    max_features=1000 
)

# Traducimos los textos a matrices

X_train_tfidf = vectorizador.fit_transform(X_train_texto)
X_test_tfidf = vectorizador.transform(X_test_texto)


# 4) REDUCCIÓN DE DIMENSIONALIDAD CON UMAP
# Reducimos las columnas a solo 10 principales
# init="random" solucion para evitar colapso de memoria
reductor_umap = umap.UMAP(
    n_components=10, 
    n_neighbors=15, 
    min_dist=0.0, 
    random_state=SEED,
    init="random"
) 

# El reductor aprende solo de X_train_tfidf
X_train_umap = reductor_umap.fit_transform(X_train_tfidf)
X_test_umap = reductor_umap.transform(X_test_tfidf)

# Con HDBSCAN busca grupos (clústeres) de reseñas que hablen de temas parecidos
# Cada grupo debe tener al menos 30 reseñas para ser considerado válido
agrupador = HDBSCAN(min_cluster_size=30)
etiquetas_clusters = agrupador.fit_predict(X_train_umap)

# Calculamos cuantos clústeres encontro y cuantas reseñas no encajan en nada
ruido = (etiquetas_clusters == -1).mean() # no encajan en nada -> ruido = etiqueta -1 
numero_grupos = len(set(etiquetas_clusters) - {-1})
print(f"HDBSCAN detectó {numero_grupos} grupos de temas distintos.")
print(f"El {ruido:.0%} de las reseñas son ruido (no encajan en ningún tema principal).")


# 5) EXPORTAMOS LOS RESULTADOS
# Exportar las matrices a un archivo comprimido para luego cargarlas
np.savez("resultados/datos_procesados.npz", 
         X_train=X_train_umap, 
         X_test=X_test_umap, 
         y_train=y_train, 
         y_test=y_test)
print("¡Archivos numéricos exportados con éxito a la carpeta resultados! - Todo listo para la fase 2!")






